try:
    import polars as pl
except ModuleNotFoundError:
    POLARS_AVAILABLE = False
    pl = object  # type: ignore[assignment]
else:
    POLARS_AVAILABLE = True
try:
    import pandas as pd
except ModuleNotFoundError:
    PANDAS_AVAILABLE = False
    pd = object
else:
    PANDAS_AVAILABLE = True
try:
    import cudf
except ModuleNotFoundError:
    CUDF_AVAILABLE = False
    cudf = object
else:
    CUDF_AVAILABLE = True
try:
    import modin.pandas as mpd
except ModuleNotFoundError:
    MODIN_AVAILABLE = False
    mpd = object
else:
    MODIN_AVAILABLE = True


from typing import Any


def is_lazyframe(obj: Any) -> bool:
    if hasattr(obj, "__lazyframe_namespace__"):
        return True
    if POLARS_AVAILABLE and isinstance(
        obj, (pl.DataFrame, pl.LazyFrame, pl.Expr, pl.Series)
    ):
        return isinstance(obj, pl.LazyFrame)
    return False


def is_dataframe(obj: Any) -> bool:
    if hasattr(obj, "__dataframe_namespace__"):
        return True
    if POLARS_AVAILABLE and isinstance(
        obj, (pl.DataFrame, pl.LazyFrame, pl.Expr, pl.Series)
    ):
        return isinstance(obj, pl.DataFrame)
    return False


def is_expr(obj: Any) -> bool:
    if hasattr(obj, "__expr_namespace__"):
        return True
    if POLARS_AVAILABLE and isinstance(
        obj, (pl.DataFrame, pl.LazyFrame, pl.Expr, pl.Series)
    ):
        return isinstance(obj, pl.Expr)
    return False


def is_series(obj: Any) -> bool:
    if hasattr(obj, "__series_namespace__"):
        return True
    if POLARS_AVAILABLE and isinstance(
        obj, (pl.DataFrame, pl.LazyFrame, pl.Expr, pl.Series)
    ):
        return isinstance(obj, pl.Series)
    return False


def get_implementation(obj: Any) -> str:
    if POLARS_AVAILABLE and isinstance(
        obj, (pl.DataFrame, pl.LazyFrame, pl.Expr, pl.Series)
    ):
        return "polars"
    if PANDAS_AVAILABLE and isinstance(obj, (pd.DataFrame, pd.Series)):
        return "pandas"
    if CUDF_AVAILABLE and isinstance(obj, (cudf.DataFrame, cudf.Series)):
        return "cudf"
    if MODIN_AVAILABLE and isinstance(obj, mpd.DataFrame):
        return "modin"
    msg = f"Unknown implementation: {obj}"
    raise TypeError(msg)


def is_pandas(obj: Any) -> bool:
    return get_implementation(obj) == "pandas"


def is_polars(obj: Any) -> bool:
    return get_implementation(obj) == "polars"


def is_cudf(obj: Any) -> bool:
    return get_implementation(obj) == "cudf"


def is_modin(obj: Any) -> bool:
    return get_implementation(obj) == "modin"
