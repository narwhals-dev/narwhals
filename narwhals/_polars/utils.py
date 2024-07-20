from narwhals.dependencies import get_polars

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