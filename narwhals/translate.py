from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import overload

if TYPE_CHECKING:
    from narwhals.spec import DataFrame
    from narwhals.spec import LazyFrame
    from narwhals.spec import Namespace


@overload
def translate_frame(
    df: Any, version: str, *, eager: Literal[True]
) -> tuple[DataFrame, Namespace]:
    ...


@overload
def translate_frame(
    df: Any, version: str, *, eager: Literal[False] = ...
) -> tuple[LazyFrame, Namespace]:
    ...


def translate_frame(
    df: Any, version: str, *, eager: bool = False
) -> tuple[DataFrame | LazyFrame, Namespace]:
    if hasattr(df, "__narwhals_frame__"):
        return df.__narwhals_frame__(version=version, eager=eager)  # type: ignore[no-any-return]
    try:
        import polars as pl
    except ModuleNotFoundError:
        pass
    else:
        if isinstance(df, pl.DataFrame) and not eager:
            msg = (
                "Expected LazyFrame, got DataFrame. Set `eager=False` if you function requires "
                "eager execution, or make you frame lazy before passing it to this function."
            )
            raise TypeError(msg)
        if isinstance(df, pl.LazyFrame) and eager:
            msg = (
                "Expected DataFrame, got LazyFrame. Set `eager=True` if you function doesn't "
                "require eager execution, or make you frame lazy before passing it to this "
                "function."
            )
            raise TypeError(msg)
        if isinstance(df, pl.DataFrame):
            return df, pl  # type: ignore[return-value]
        if isinstance(df, pl.LazyFrame) and not eager:
            return df, pl  # type: ignore[return-value]
    try:
        import pandas as pd
    except ModuleNotFoundError:
        pass
    else:
        if isinstance(df, pd.DataFrame):
            from narwhals.pandas_like.translate import translate

            return translate(
                df, api_version=version, implementation="pandas", eager=eager
            )
    try:
        import cudf
    except ModuleNotFoundError:
        pass
    else:
        if isinstance(df, cudf.DataFrame):
            from narwhals.pandas_like.translate import translate

            return translate(df, api_version=version, implementation="cudf", eager=eager)
    try:
        import modin.pandas as mpd
    except ModuleNotFoundError:
        pass
    else:
        if isinstance(df, mpd.DataFrame):
            from narwhals.pandas_like.translate import translate

            return translate(df, api_version=version, implementation="modin", eager=eager)
    msg = f"Could not translate DataFrame {type(df)}, please open a feature request."
    raise TypeError(msg)


def to_original_object(df: DataFrame | LazyFrame) -> Any:
    try:
        import polars as pl
    except ModuleNotFoundError:
        pass
    else:
        if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
            return df
    return df._dataframe  # type: ignore[union-attr]


def get_namespace(obj: Any, implementation: str | None = None) -> Namespace:
    if implementation == "polars":
        import polars as pl

        return pl  # type: ignore[return-value]
    try:
        import polars as pl
    except ModuleNotFoundError:
        pass
    else:
        if isinstance(obj, (pl.DataFrame, pl.LazyFrame, pl.Series)):
            return pl  # type: ignore[return-value]
    from narwhals.pandas_like.namespace import Namespace

    return Namespace(api_version="0.20.0", implementation=obj._implementation)
