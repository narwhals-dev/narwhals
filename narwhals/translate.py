from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import overload

from narwhals.dependencies import get_cudf
from narwhals.dependencies import get_modin
from narwhals.dependencies import get_pandas
from narwhals.dependencies import get_polars

if TYPE_CHECKING:
    from narwhals.spec import DataFrame
    from narwhals.spec import LazyFrame
    from narwhals.spec import Namespace


@overload
def translate_frame(
    df: Any,
    *,
    eager_only: Literal[False] = ...,
    lazy_only: Literal[False] = ...,
) -> tuple[DataFrame | LazyFrame, Namespace]:
    ...


@overload
def translate_frame(
    df: Any,
    *,
    eager_only: Literal[True],
    lazy_only: Literal[False] = ...,
) -> tuple[DataFrame, Namespace]:
    ...


@overload
def translate_frame(
    df: Any,
    *,
    eager_only: Literal[False] = ...,
    lazy_only: Literal[True],
) -> tuple[LazyFrame, Namespace]:
    ...


def translate_frame(
    df: Any,
    *,
    eager_only: bool = False,
    lazy_only: bool = False,
) -> tuple[DataFrame | LazyFrame, Namespace]:
    if eager_only and lazy_only:
        msg = "Only one of `eager_only` and `lazy_only` can be True."
        raise ValueError(msg)

    if hasattr(df, "__narwhals_frame__"):
        return df.__narwhals_frame__(eager_only=eager_only, lazy_only=lazy_only)  # type: ignore[no-any-return]

    if (pl := get_polars()) is not None:
        if isinstance(df, pl.LazyFrame) and eager_only:
            msg = (
                "Expected DataFrame, got LazyFrame. Set `eager_only=False` if you "
                "function doesn't require eager execution, or collect your frame "
                "before passing it to this function."
            )
            raise TypeError(msg)
        if isinstance(df, pl.DataFrame) and lazy_only:
            msg = (
                "Expected LazyFrame, got DataFrame. Set `lazy_only=False` if you "
                "function doesn't doesn't need to use `.collect`, or make your frame "
                "before passing it to this function."
            )
            raise TypeError(msg)
        if isinstance(df, pl.DataFrame):
            from narwhals.polars import DataFrame
            from narwhals.polars import Namespace

            return DataFrame(df), Namespace()
        if isinstance(df, pl.LazyFrame):
            from narwhals.polars import LazyFrame
            from narwhals.polars import Namespace

            return LazyFrame(df), Namespace()

    if (pd := get_pandas()) is not None and isinstance(df, pd.DataFrame):
        from narwhals.pandas_like.translate import translate

        return translate(
            df,
            implementation="pandas",
            eager_only=eager_only,
            lazy_only=lazy_only,
        )

    if (cudf := get_cudf()) is not None and isinstance(df, cudf.DataFrame):
        from narwhals.pandas_like.translate import translate

        return translate(
            df, implementation="cudf", eager_only=eager_only, lazy_only=lazy_only
        )

    if (mpd := get_modin()) is not None and isinstance(df, mpd.DataFrame):
        from narwhals.pandas_like.translate import translate

        return translate(
            df, implementation="modin", eager_only=eager_only, lazy_only=lazy_only
        )

    msg = f"Could not translate DataFrame {type(df)}, please open a feature request."
    raise TypeError(msg)


def to_original_object(df: DataFrame | LazyFrame) -> Any:
    if (pl := get_polars()) is not None and isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        return df
    return df._dataframe  # type: ignore[union-attr]
