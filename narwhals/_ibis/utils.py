from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import Mapping
from typing import Sequence

import ibis
import ibis.expr.datatypes as ibis_dtypes

from narwhals.utils import isinstance_or_issubclass

if TYPE_CHECKING:
    import ibis.expr.types as ir
    from ibis.expr.datatypes import DataType as IbisDataType
    from typing_extensions import TypeAlias
    from typing_extensions import TypeIs

    from narwhals._duration import IntervalUnit
    from narwhals._ibis.dataframe import IbisLazyFrame
    from narwhals._ibis.expr import IbisExpr
    from narwhals.dtypes import DType
    from narwhals.utils import Version

lit = ibis.literal
"""Alias for `ibis.literal`."""

BucketUnit: TypeAlias = Literal[
    "years",
    "quarters",
    "months",
    "days",
    "hours",
    "minutes",
    "seconds",
    "milliseconds",
    "microseconds",
    "nanoseconds",
]
TruncateUnit: TypeAlias = Literal[
    "Y", "Q", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"
]

UNITS_DICT_BUCKET: Mapping[IntervalUnit, BucketUnit] = {
    "y": "years",
    "q": "quarters",
    "mo": "months",
    "d": "days",
    "h": "hours",
    "m": "minutes",
    "s": "seconds",
    "ms": "milliseconds",
    "us": "microseconds",
    "ns": "nanoseconds",
}

UNITS_DICT_TRUNCATE: Mapping[IntervalUnit, TruncateUnit] = {
    "y": "Y",
    "q": "Q",
    "mo": "M",
    "d": "D",
    "h": "h",
    "m": "m",
    "s": "s",
    "ms": "ms",
    "us": "us",
    "ns": "ns",
}


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
        native_series_list = expr(df)
        output_names = expr._evaluate_output_names(df)
        if expr._alias_output_names is not None:
            output_names = expr._alias_output_names(output_names)
        if len(output_names) != len(native_series_list):  # pragma: no cover
            msg = f"Internal error: got output names {output_names}, but only got {len(native_series_list)} results"
            raise AssertionError(msg)
        native_results.extend(zip(output_names, native_series_list))
    return native_results


@lru_cache(maxsize=16)
def native_to_narwhals_dtype(ibis_dtype: IbisDataType, version: Version) -> DType:  # noqa: C901, PLR0912
    dtypes = version.dtypes
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
    if is_interval(ibis_dtype):
        _time_unit = ibis_dtype.unit.value
        if _time_unit not in {"ns", "us", "ms", "s"}:  # pragma: no cover
            msg = f"Unsupported interval unit: {_time_unit}"
            raise NotImplementedError(msg)
        return dtypes.Duration(_time_unit)
    if is_array(ibis_dtype):
        if ibis_dtype.length:
            return dtypes.Array(
                native_to_narwhals_dtype(ibis_dtype.value_type, version),
                ibis_dtype.length,
            )
        else:
            return dtypes.List(native_to_narwhals_dtype(ibis_dtype.value_type, version))
    if is_struct(ibis_dtype):
        return dtypes.Struct(
            [
                dtypes.Field(name, native_to_narwhals_dtype(dtype, version))
                for name, dtype in ibis_dtype.items()
            ]
        )
    if ibis_dtype.is_decimal():  # pragma: no cover
        return dtypes.Decimal()
    if ibis_dtype.is_time():
        return dtypes.Time()
    if ibis_dtype.is_binary():
        return dtypes.Binary()
    return dtypes.Unknown()  # pragma: no cover


def is_interval(obj: IbisDataType) -> TypeIs[ibis_dtypes.Interval]:
    return obj.is_interval()


def is_array(obj: IbisDataType) -> TypeIs[ibis_dtypes.Array[Any]]:
    return obj.is_array()


def is_struct(obj: IbisDataType) -> TypeIs[ibis_dtypes.Struct]:
    return obj.is_struct()


def is_floating(obj: IbisDataType) -> TypeIs[ibis_dtypes.Floating]:
    return obj.is_floating()


def narwhals_to_native_dtype(  # noqa: C901, PLR0912
    dtype: DType | type[DType], version: Version
) -> IbisDataType:
    dtypes = version.dtypes

    if isinstance_or_issubclass(dtype, dtypes.Decimal):  # pragma: no cover
        return ibis_dtypes.Decimal()
    if isinstance_or_issubclass(dtype, dtypes.Float64):
        return ibis_dtypes.Float64()
    if isinstance_or_issubclass(dtype, dtypes.Float32):
        return ibis_dtypes.Float32()
    if isinstance_or_issubclass(dtype, dtypes.Int128):  # pragma: no cover
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
    if isinstance_or_issubclass(dtype, dtypes.UInt128):  # pragma: no cover
        msg = "UInt128 not supported by Ibis"
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.UInt64):
        return ibis_dtypes.UInt64()
    if isinstance_or_issubclass(dtype, dtypes.UInt32):
        return ibis_dtypes.UInt32()
    if isinstance_or_issubclass(dtype, dtypes.UInt16):
        return ibis_dtypes.UInt16()
    if isinstance_or_issubclass(dtype, dtypes.UInt8):
        return ibis_dtypes.UInt8()
    if isinstance_or_issubclass(dtype, dtypes.String):
        return ibis_dtypes.String()
    if isinstance_or_issubclass(dtype, dtypes.Boolean):
        return ibis_dtypes.Boolean()
    if isinstance_or_issubclass(dtype, dtypes.Categorical):
        msg = "Categorical not supported by Ibis"
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Datetime):
        return ibis_dtypes.Timestamp()
    if isinstance_or_issubclass(dtype, dtypes.Duration):
        return ibis_dtypes.Interval(unit=dtype.time_unit)  # pyright: ignore[reportArgumentType]
    if isinstance_or_issubclass(dtype, dtypes.Date):
        return ibis_dtypes.Date()
    if isinstance_or_issubclass(dtype, dtypes.Time):
        return ibis_dtypes.Time()
    if isinstance_or_issubclass(dtype, dtypes.List):
        inner = narwhals_to_native_dtype(dtype.inner, version)
        return ibis_dtypes.Array(value_type=inner)
    if isinstance_or_issubclass(dtype, dtypes.Struct):
        fields = [
            (field.name, narwhals_to_native_dtype(field.dtype, version))
            for field in dtype.fields
        ]
        return ibis_dtypes.Struct.from_tuples(fields)
    if isinstance_or_issubclass(dtype, dtypes.Array):
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
