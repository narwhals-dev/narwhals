from __future__ import annotations

from typing import TYPE_CHECKING

import daft
import daft.datatype
from daft import DataType

from narwhals._utils import isinstance_or_issubclass

if TYPE_CHECKING:
    from narwhals._daft.dataframe import DaftLazyFrame
    from narwhals._daft.expr import DaftExpr
    from narwhals._utils import Version
    from narwhals.dtypes import DType

lit = daft.lit
"""Alias for `daft.lit`."""


def maybe_evaluate_expr(df: DaftLazyFrame, obj: DaftExpr | object) -> daft.Expression:
    from narwhals._daft.expr import DaftExpr

    if isinstance(obj, DaftExpr):
        column_results = obj._call(df)
        if len(column_results) != 1:
            msg = "Multi-output expressions (e.g. `nw.all()` or `nw.col('a', 'b')`) not supported in this context"
            raise ValueError(msg)
        return column_results[0]
    return daft.lit(obj)


def evaluate_exprs(
    df: DaftLazyFrame, /, *exprs: DaftExpr
) -> list[tuple[str, daft.Expression]]:
    native_results: list[tuple[str, daft.Expression]] = []
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


def native_to_narwhals_dtype(daft_dtype: DataType, version: Version) -> DType:  # noqa: PLR0912,C901
    dtypes = version.dtypes

    if daft_dtype == DataType.int64():
        return dtypes.Int64()
    if daft_dtype == DataType.int32():
        return dtypes.Int32()
    if daft_dtype == DataType.int16():
        return dtypes.Int16()
    if daft_dtype == DataType.int8():
        return dtypes.Int8()
    if daft_dtype == DataType.uint64():
        return dtypes.UInt64()
    if daft_dtype == DataType.uint32():
        return dtypes.UInt32()
    if daft_dtype == DataType.uint16():
        return dtypes.UInt16()
    if daft_dtype == DataType.uint8():
        return dtypes.UInt8()
    if daft_dtype == DataType.float64():
        return dtypes.Float64()
    if daft_dtype == DataType.float32():
        return dtypes.Float32()
    if daft_dtype == DataType.string():
        return dtypes.String()
    if daft_dtype == DataType.date():
        return dtypes.Date()
    if daft_dtype == DataType.timestamp("us", None):
        return dtypes.Datetime("us", None)
    if daft_dtype == DataType.bool():
        return dtypes.Boolean()
    if daft_dtype == DataType.duration("us"):
        return dtypes.Duration("us")
    if daft_dtype == DataType.decimal128(1, 1):
        return dtypes.Decimal()
    if DataType.is_fixed_size_list(daft_dtype):
        return dtypes.Array(
            native_to_narwhals_dtype(daft_dtype.dtype, version), daft_dtype.size
        )
    return dtypes.Unknown()  # pragma: no cover


def narwhals_to_native_dtype(  # noqa: PLR0912,C901
    dtype: DType | type[DType], version: Version, backend_version: tuple[int, ...]
) -> daft.DataType:
    dtypes = version.dtypes
    if dtype == dtypes.Float64:
        return DataType.float64()
    if dtype == dtypes.Float32:
        return DataType.float32()
    if dtype in {dtypes.Int128, dtypes.UInt128}:
        msg = "Converting to Int128/UInt128 is not (yet) supported for Daft."
        raise NotImplementedError(msg)
    if dtype == dtypes.Int64:
        return DataType.int64()
    if dtype == dtypes.Int32:
        return DataType.int32()
    if dtype == dtypes.Int16:
        return DataType.int16()
    if dtype == dtypes.Int8:
        return DataType.int8()
    if dtype == dtypes.UInt64:
        return DataType.uint64()
    if dtype == dtypes.UInt32:
        return DataType.uint32()
    if dtype == dtypes.UInt16:
        return DataType.uint16()
    if dtype == dtypes.UInt8:
        return DataType.uint8()
    if dtype == dtypes.String:
        return DataType.string()
    if dtype == dtypes.Boolean:
        return DataType.bool()
    if dtype == dtypes.Object:  # pragma: no cover
        msg = "Converting to Object is not (yet) supported for Daft"
        raise NotImplementedError(msg)
    if dtype == dtypes.Categorical:
        msg = "Converting to Categorical is not (yet) supported for Daft"
        raise NotImplementedError(msg)
    if dtype == dtypes.Enum:
        msg = "Converting to Enum is not (yet) supported for Daft"
        raise NotImplementedError(msg)
    if dtype == dtypes.Date:
        return DataType.date()
    if dtype == dtypes.Time:
        return DataType.time("ns")
    if dtype == dtypes.Binary:
        return DataType.binary()
    if dtype == dtypes.Decimal:
        msg = "Casting to Decimal is not supported yet."
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Datetime):
        return DataType.timestamp(dtype.time_unit, dtype.time_zone)
    if isinstance_or_issubclass(dtype, dtypes.Duration):
        return DataType.duration(dtype.time_unit)
    if isinstance_or_issubclass(dtype, dtypes.List):
        return DataType.list(
            narwhals_to_native_dtype(dtype.inner, version, backend_version)
        )
    if isinstance_or_issubclass(dtype, dtypes.Struct):
        return DataType.struct(
            {
                field.name: narwhals_to_native_dtype(
                    field.dtype, version, backend_version
                )
                for field in dtype.fields
            }
        )
    if isinstance_or_issubclass(dtype, dtypes.Array):  # pragma: no cover
        return DataType.fixed_size_list(
            narwhals_to_native_dtype(dtype.inner, version, backend_version), dtype.size
        )
    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)
