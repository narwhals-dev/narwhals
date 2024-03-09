from typing import Any

from narwhals.translate import get_pandas
from narwhals.translate import get_polars


def is_dataframe(obj: Any) -> bool:
    if (pl := get_polars()) is not None and isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        return True
    if (pd := get_pandas()) is not None and isinstance(obj, pd.DataFrame):
        return True
    return False


def is_series(obj: Any) -> bool:
    if (pl := get_polars()) is not None and isinstance(obj, (pl.Series)):
        return True
    if (pd := get_pandas()) is not None and isinstance(obj, pd.Series):
        return True
    raise NotImplementedError


def get_implementation(obj: Any) -> str:
    if (pl := get_polars()) is not None and isinstance(
        obj, (pl.DataFrame, pl.LazyFrame, pl.Expr, pl.Series)
    ):
        return "polars"
    if (pd := get_pandas()) is not None and isinstance(obj, (pd.DataFrame, pd.Series)):
        return "pandas"
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
