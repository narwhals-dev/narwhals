from narwhals.dependencies import get_polars
from narwhals import dtypes

def extract_native(obj):
    from narwhals._polars.dataframe import PolarsDataFrame, PolarsLazyFrame
    from narwhals._polars.series import PolarsSeries
    from narwhals._polars.expr import PolarsExpr

    if isinstance(obj, (PolarsDataFrame, PolarsLazyFrame)):
        return obj._native_dataframe
    if isinstance(obj, PolarsSeries):
        return obj._native_series
    if isinstance(obj, PolarsExpr):
        return obj._native_expr
    return obj

def extract_args_kwargs(args, kwargs):
    args = [extract_native(arg) for arg in args]
    kwargs = {k: extract_native(v) for k, v in kwargs.items()}
    return args, kwargs

def translate_dtype(dtype):
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
