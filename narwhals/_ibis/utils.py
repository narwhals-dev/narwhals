from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

import ibis

from narwhals.utils import import_dtypes_module
from narwhals.utils import isinstance_or_issubclass

if TYPE_CHECKING:
    import ibis.expr.types as ir
    from ibis.expr.datatypes import DataType as IbisDataType

    from narwhals._ibis.dataframe import IbisLazyFrame
    from narwhals._ibis.expr import IbisExpr
    from narwhals.dtypes import DType
    from narwhals.utils import Version

lit = ibis.literal
"""Alias for `ibis.literal`."""


class WindowInputs:
    __slots__ = ("expr", "order_by", "partition_by")

    def __init__(
        self,
        expr: ir.Expr | ir.Value | ir.Column,
        partition_by: Sequence[str],
        order_by: Sequence[str],
    ) -> None:
        self.expr = expr
        self.partition_by = partition_by
        self.order_by = order_by


def evaluate_exprs(df: IbisLazyFrame, /, *exprs: IbisExpr) -> list[tuple[str, ir.Value]]:
    native_results: list[tuple[str, ir.Value]] = []
    for expr in exprs:
        native_series_list = expr._call(df)
        output_names = expr._evaluate_output_names(df)
        if expr._alias_output_names is not None:
            output_names = expr._alias_output_names(output_names)
        if len(output_names) != len(native_series_list):  # pragma: no cover
            msg = f"Internal error: got output names {output_names}, but only got {len(native_series_list)} results"
            raise AssertionError(msg)
        native_results.extend(zip(output_names, native_series_list))
    return native_results


@lru_cache(maxsize=16)
def native_to_narwhals_dtype(ibis_dtype: Any, version: Version) -> DType:
    dtypes = import_dtypes_module(version)
    if ibis_dtype.is_int64():
        return dtypes.Int64()
    if ibis_dtype.is_int32():
        return dtypes.Int32()
    if ibis_dtype.is_int16():
        return dtypes.Int16()
    if ibis_dtype.is_int8():
        return dtypes.Int8()
    if ibis_dtype.is_uint64():
        return dtypes.UInt64()
    if ibis_dtype.is_uint32():
        return dtypes.UInt32()
    if ibis_dtype.is_uint16():
        return dtypes.UInt16()
    if ibis_dtype.is_uint8():
        return dtypes.UInt8()
    if ibis_dtype.is_boolean():
        return dtypes.Boolean()
    if ibis_dtype.is_float64():
        return dtypes.Float64()
    if ibis_dtype.is_float32():
        return dtypes.Float32()
    if ibis_dtype.is_string():
        return dtypes.String()
    if ibis_dtype.is_date():
        return dtypes.Date()
    if ibis_dtype.is_timestamp():
        return dtypes.Datetime()
    if ibis_dtype.is_interval():
        _time_unit = ibis_dtype.unit
        if _time_unit not in {"ns", "us", "ms", "s"}:
            msg = f"Unsupported interval unit: {_time_unit}"
            raise NotImplementedError(msg)
        return dtypes.Duration(_time_unit)
    if ibis_dtype.is_array():
        if ibis_dtype.length:
            return dtypes.Array(
                native_to_narwhals_dtype(ibis_dtype.value_type, version),
                ibis_dtype.length,
            )
        else:
            return dtypes.List(native_to_narwhals_dtype(ibis_dtype.value_type, version))
    if ibis_dtype.is_struct():
        return dtypes.Struct(
            [
                dtypes.Field(
                    ibis_dtype_name,
                    native_to_narwhals_dtype(ibis_dtype_field, version),
                )
                for ibis_dtype_name, ibis_dtype_field in ibis_dtype.items()
            ]
        )
    if ibis_dtype.is_decimal():
        return dtypes.Decimal()
    if ibis_dtype.is_time():
        return dtypes.Time()
    if ibis_dtype.is_binary():
        return dtypes.Binary()
    return dtypes.Unknown()  # pragma: no cover


def narwhals_to_native_dtype(
    dtype: DType | type[DType], version: Version
) -> IbisDataType:
    dtypes = import_dtypes_module(version)
    ibis_dtypes = ibis.expr.datatypes

    if isinstance_or_issubclass(dtype, dtypes.Decimal):
        return ibis_dtypes.Decimal()
    if isinstance_or_issubclass(dtype, dtypes.Float64):
        return ibis_dtypes.Float64()
    if isinstance_or_issubclass(dtype, dtypes.Float32):
        return ibis_dtypes.Float32()
    if isinstance_or_issubclass(dtype, dtypes.Int128):
        msg = "Int128 not supported by Ibis"
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Int64):
        return ibis_dtypes.Int64()
    if isinstance_or_issubclass(dtype, dtypes.Int32):
        return ibis_dtypes.Int32()
    if isinstance_or_issubclass(dtype, dtypes.Int16):
        return ibis_dtypes.Int16()
    if isinstance_or_issubclass(dtype, dtypes.Int8):
        return ibis_dtypes.Int8()
    if isinstance_or_issubclass(dtype, dtypes.UInt128):
        msg = "UInt128 not supported by Ibis"
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.UInt64):
        return ibis_dtypes.UInt64()
    if isinstance_or_issubclass(dtype, dtypes.UInt32):
        return ibis_dtypes.UInt32()
    if isinstance_or_issubclass(dtype, dtypes.UInt16):  # pragma: no cover
        return ibis_dtypes.UInt16()
    if isinstance_or_issubclass(dtype, dtypes.UInt8):  # pragma: no cover
        return ibis_dtypes.UInt8()
    if isinstance_or_issubclass(dtype, dtypes.String):
        return ibis_dtypes.String()
    if isinstance_or_issubclass(dtype, dtypes.Boolean):  # pragma: no cover
        return ibis_dtypes.Boolean()
    if isinstance_or_issubclass(dtype, dtypes.Categorical):
        msg = "Categorical not supported by Ibis"
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Datetime):
        return ibis_dtypes.Timestamp()
    if isinstance_or_issubclass(dtype, dtypes.Duration):  # pragma: no cover
        _time_unit = getattr(dtype, "time_unit", "us")
        return ibis_dtypes.Interval(_time_unit)
    if isinstance_or_issubclass(dtype, dtypes.Date):  # pragma: no cover
        return ibis_dtypes.Date()
    if isinstance_or_issubclass(dtype, dtypes.Time):
        return ibis_dtypes.Time()
    if isinstance_or_issubclass(dtype, dtypes.List):
        inner = narwhals_to_native_dtype(dtype.inner, version)
        return ibis_dtypes.Array(value_type=inner)
    if isinstance_or_issubclass(dtype, dtypes.Struct):  # pragma: no cover
        fields = [
            (field.name, narwhals_to_native_dtype(field.dtype, version))
            for field in dtype.fields
        ]
        return ibis_dtypes.Struct.from_tuples(fields)
    if isinstance_or_issubclass(dtype, dtypes.Array):  # pragma: no cover
        inner = narwhals_to_native_dtype(dtype.inner, version)
        return ibis_dtypes.Array(value_type=inner, length=dtype.size)
    if isinstance_or_issubclass(dtype, dtypes.Binary):
        return ibis_dtypes.Binary()
    if isinstance_or_issubclass(dtype, dtypes.Enum):
        # Ibis does not support: https://github.com/ibis-project/ibis/issues/10991
        msg = "Enum not supported by Ibis"
        raise NotImplementedError(msg)
    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)
