from __future__ import annotations

from enum import Enum
from enum import auto
from functools import lru_cache
from typing import TYPE_CHECKING
from typing import Any

from narwhals.utils import import_dtypes_module
from narwhals.utils import isinstance_or_issubclass

if TYPE_CHECKING:
    import ibis.expr.types as ir

    from narwhals._ibis.dataframe import IbisLazyFrame
    from narwhals._ibis.expr import IbisExpr
    from narwhals.dtypes import DType
    from narwhals.utils import Version


class ExprKind(Enum):
    """Describe which kind of expression we are dealing with.

    Composition rule is:
    - LITERAL vs LITERAL -> LITERAL
    - TRANSFORM vs anything -> TRANSFORM
    - anything vs TRANSFORM -> TRANSFORM
    - all remaining cases -> AGGREGATION
    """

    LITERAL = auto()  # e.g. nw.lit(1)
    AGGREGATION = auto()  # e.g. nw.col('a').mean()
    TRANSFORM = auto()  # e.g. nw.col('a').round()


def maybe_evaluate(df: IbisLazyFrame, obj: Any) -> Any:
    from narwhals._ibis.expr import IbisExpr

    if isinstance(obj, IbisExpr):
        column_results = obj._call(df)
        if len(column_results) != 1:  # pragma: no cover
            msg = "Multi-output expressions (e.g. `nw.all()` or `nw.col('a', 'b')`) not supported in this context"
            raise NotImplementedError(msg)
        return column_results[0]
    return obj


def parse_exprs(df: IbisLazyFrame, /, *exprs: IbisExpr) -> dict[str, ir.Expr]:
    native_results: dict[str, ir.Expr] = {}
    for expr in exprs:
        native_series_list = expr._call(df)
        output_names = expr._evaluate_output_names(df)
        if expr._alias_output_names is not None:
            output_names = expr._alias_output_names(output_names)
        if len(output_names) != len(native_series_list):  # pragma: no cover
            msg = f"Internal error: got output names {output_names}, but only got {len(native_series_list)} results"
            raise AssertionError(msg)
        native_results.update(zip(output_names, native_series_list))
    return native_results


@lru_cache(maxsize=16)
def native_to_narwhals_dtype(dtype: Any, version: Version) -> DType:
    dtypes = import_dtypes_module(version)

    if dtype.is_int64():
        return dtypes.Int64()
    if dtype.is_int32():
        return dtypes.Int32()
    if dtype.is_int16():
        return dtypes.Int16()
    if dtype.is_int8():
        return dtypes.Int8()
    if dtype.is_uint64():
        return dtypes.UInt64()
    if dtype.is_uint32():
        return dtypes.UInt32()
    if dtype.is_uint16():
        return dtypes.UInt16()
    if dtype.is_uint8():
        return dtypes.UInt8()
    if dtype.is_boolean():
        return dtypes.Boolean()
    if dtype.is_float64():
        return dtypes.Float64()
    if dtype.is_float32():
        return dtypes.Float32()
    if dtype.is_string():
        return dtypes.String()
    if dtype.is_date():
        return dtypes.Date()
    if dtype.is_timestamp():
        return dtypes.Datetime(time_zone=dtype.timezone, time_unit=dtype.unit.value)
    if dtype.is_array():
        return dtypes.List(native_to_narwhals_dtype(dtype.value_type, version))
    if dtype.is_struct():
        return dtypes.Struct(
            [
                dtypes.Field(
                    ibis_dtype_name,
                    native_to_narwhals_dtype(ibis_dtype_field, version),
                )
                for ibis_dtype_name, ibis_dtype_field in dtype.items()
            ]
        )
    if dtype.is_decimal():
        return dtypes.Decimal()
    return dtypes.Unknown()  # pragma: no cover


def narwhals_to_native_dtype(dtype: DType | type[DType], version: Version) -> str:
    dtypes = import_dtypes_module(version)
    if isinstance_or_issubclass(dtype, dtypes.Decimal):
        return "decimal"
    if isinstance_or_issubclass(dtype, dtypes.Float64):
        return "float64"
    if isinstance_or_issubclass(dtype, dtypes.Float32):
        return "float32"
    if isinstance_or_issubclass(dtype, dtypes.Int128):
        msg = "Int128 not supported by Ibis"
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Int64):
        return "int64"
    if isinstance_or_issubclass(dtype, dtypes.Int32):
        return "int32"
    if isinstance_or_issubclass(dtype, dtypes.Int16):
        return "int16"
    if isinstance_or_issubclass(dtype, dtypes.Int8):
        return "int8"
    if isinstance_or_issubclass(dtype, dtypes.UInt128):
        msg = "UInt128 not supported by Ibis"
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.UInt64):
        return "uint64"
    if isinstance_or_issubclass(dtype, dtypes.UInt32):
        return "uint32"
    if isinstance_or_issubclass(dtype, dtypes.UInt16):  # pragma: no cover
        return "uint16"
    if isinstance_or_issubclass(dtype, dtypes.UInt8):  # pragma: no cover
        return "uint8"
    if isinstance_or_issubclass(dtype, dtypes.String):
        return "string"
    if isinstance_or_issubclass(dtype, dtypes.Boolean):  # pragma: no cover
        return "bool"
    if isinstance_or_issubclass(dtype, dtypes.Categorical):
        msg = "Categorical not supported by Ibis"
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Datetime):
        return "timestamp"
    if isinstance_or_issubclass(dtype, dtypes.Duration):  # pragma: no cover
        _time_unit = getattr(dtype, "time_unit", "us")
        msg = "Categorical not supported by Ibis"
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Date):  # pragma: no cover
        return "date"
    if isinstance_or_issubclass(dtype, dtypes.List):
        inner = narwhals_to_native_dtype(dtype.inner, version)  # type: ignore[union-attr]
        return f"array<{inner}>"
    if isinstance_or_issubclass(dtype, dtypes.Struct):  # pragma: no cover
        inner = ", ".join(
            f"{field.name}: {narwhals_to_native_dtype(field.dtype, version)}"
            for field in dtype.fields  # type: ignore[union-attr]
        )
        return f"struct<{inner}>"
    if isinstance_or_issubclass(dtype, dtypes.Array):  # pragma: no cover
        inner = narwhals_to_native_dtype(dtype.inner, version)  # type: ignore[union-attr]
        return f"array<{inner}>"
    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def n_ary_operation_expr_kind(*args: IbisExpr | Any) -> ExprKind:
    if all(
        getattr(arg, "_expr_kind", ExprKind.LITERAL) is ExprKind.LITERAL for arg in args
    ):
        return ExprKind.LITERAL
    if any(
        getattr(arg, "_expr_kind", ExprKind.LITERAL) is ExprKind.TRANSFORM for arg in args
    ):
        return ExprKind.TRANSFORM
    return ExprKind.AGGREGATION
