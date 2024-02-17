from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from polars_api_compat.spec import DataFrame
    from polars_api_compat.spec import LazyFrame
    from polars_api_compat.spec import Namespace


def to_polars_api(df: Any, version: str) -> tuple[LazyFrame, Namespace]:  # noqa: ARG001
    if hasattr(df, "__polars_api_compat__"):
        return df.__polars_api_compat__()
    try:
        import polars as pl
    except ModuleNotFoundError:
        pass
    else:
        if isinstance(df, pl.DataFrame):
            return df.lazy(), pl  # type: ignore[return-value]
        if isinstance(df, pl.LazyFrame):
            return df, pl  # type: ignore[return-value]
    try:
        import pandas as pd
    except ModuleNotFoundError:
        pass
    else:
        if isinstance(df, pd.DataFrame):
            from polars_api_compat.pandas import translate

            return translate(df)
    msg = f"Could not translate DataFrame {type(df)}, please open a feature request."
    raise TypeError(msg)


def to_native(df: DataFrame | LazyFrame) -> Any:
    try:
        import polars as pl
    except ModuleNotFoundError:
        pass
    else:
        if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
            return df
    return df.dataframe
