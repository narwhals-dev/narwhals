from __future__ import annotations

from typing import Any

from narwhals import dtypes
from narwhals.dependencies import get_polars


def extract_native(obj: Any) -> Any:
    from narwhals._polars.dataframe import PolarsDataFrame
    from narwhals._polars.dataframe import PolarsLazyFrame
    from narwhals._polars.expr import PolarsExpr
    from narwhals._polars.series import PolarsSeries

    if isinstance(obj, (PolarsDataFrame, PolarsLazyFrame)):
        return obj._native_frame
    if isinstance(obj, PolarsSeries):
        return obj._native_series
    if isinstance(obj, PolarsExpr):
        return obj._native_expr
    return obj


def extract_args_kwargs(args: Any, kwargs: Any) -> tuple[list[Any], dict[str, Any]]:
    args = [extract_native(arg) for arg in args]
    kwargs = {k: extract_native(v) for k, v in kwargs.items()}
    return args, kwargs


def translate_dtype(dtype: Any) -> dtypes.DType:
    pl = get_polars()
    if dtype == pl.Float64:
        return dtypes.Float64()
    if dtype == pl.Float32:
        return dtypes.Float32()
    if dtype == pl.Int64:
        return dtypes.Int64()
    if dtype == pl.Int32:
        return dtypes.Int32()
    if dtype == pl.Int16:
        return dtypes.Int16()
    if dtype == pl.Int8:
        return dtypes.Int8()
    if dtype == pl.UInt64:
        return dtypes.UInt64()
    if dtype == pl.UInt32:
        return dtypes.UInt32()
    if dtype == pl.UInt16:
        return dtypes.UInt16()
    if dtype == pl.UInt8:
        return dtypes.UInt8()
    if dtype == pl.String:
        return dtypes.String()
    if dtype == pl.Boolean:
        return dtypes.Boolean()
    if dtype == pl.Object:
        return dtypes.Object()
    if dtype == pl.Categorical:
        return dtypes.Categorical()
    if dtype == pl.Enum:
        return dtypes.Enum()
    if dtype == pl.Datetime:
        return dtypes.Datetime()
    if dtype == pl.Duration:
        return dtypes.Duration()
    if dtype == pl.Date:
        return dtypes.Date()
    return dtypes.Unknown()


def narwhals_to_native_dtype(dtype: dtypes.DType | type[dtypes.DType]) -> Any:
    pl = get_polars()
    from narwhals import dtypes

    if dtype == dtypes.Float64:
        return pl.Float64()
    if dtype == dtypes.Float32:
        return pl.Float32()
    if dtype == dtypes.Int64:
        return pl.Int64()
    if dtype == dtypes.Int32:
        return pl.Int32()
    if dtype == dtypes.Int16:
        return pl.Int16()
    if dtype == dtypes.Int8:
        return pl.Int8()
    if dtype == dtypes.UInt64:
        return pl.UInt64()
    if dtype == dtypes.UInt32:
        return pl.UInt32()
    if dtype == dtypes.UInt16:
        return pl.UInt16()
    if dtype == dtypes.UInt8:
        return pl.UInt8()
    if dtype == dtypes.String:
        return pl.String()
    if dtype == dtypes.Boolean:
        return pl.Boolean()
    if dtype == dtypes.Object:  # pragma: no cover
        return pl.Object()
    if dtype == dtypes.Categorical:
        return pl.Categorical()
    if dtype == dtypes.Enum:
        msg = "Converting to Enum is not (yet) supported"
        raise NotImplementedError(msg)
    if dtype == dtypes.Datetime:
        return pl.Datetime()
    if dtype == dtypes.Duration:
        return pl.Duration()
    if dtype == dtypes.Date:
        return pl.Date()
    return pl.Unknown()  # pragma: no cover


def convert_str_slice_to_int_slice(
    str_slice: slice, columns: list[str]
) -> tuple[int | None, int | None, int | None]:  # pragma: no cover
    start = columns.index(str_slice.start) if str_slice.start is not None else None
    stop = columns.index(str_slice.stop) + 1 if str_slice.stop is not None else None
    step = str_slice.step
    return (start, stop, step)
