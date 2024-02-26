from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple

from narwhals.dependencies import get_cudf
from narwhals.dependencies import get_modin
from narwhals.dependencies import get_pandas
from narwhals.dependencies import get_polars

if TYPE_CHECKING:
    from narwhals.spec import DataFrame
    from narwhals.spec import Namespace
    from narwhals.spec import Series


class FrameTranslation(NamedTuple):
    frame: DataFrame
    namespace: Namespace


class SeriesTranslation(NamedTuple):
    series: Series
    namespace: Namespace


def translate_frame(
    df: Any,
    *,
    is_eager: bool = False,
    is_lazy: bool = False,
) -> FrameTranslation:
    if is_eager and is_lazy:
        msg = "Only one of `is_eager` and `is_lazy` can be True."
        raise ValueError(msg)

    if hasattr(df, "__narwhals_frame__"):
        return FrameTranslation(
            *df.__narwhals_frame__(is_eager=is_eager, is_lazy=is_lazy)
        )

    if (pl := get_polars()) is not None:
        if isinstance(df, pl.LazyFrame) and is_eager:
            msg = (
                "Expected DataFrame, got LazyFrame. Set `is_eager=False` if you "
                "function doesn't require eager execution, or collect your frame "
                "before passing it to this function."
            )
            raise TypeError(msg)
        if isinstance(df, pl.DataFrame) and is_lazy:
            msg = (
                "Expected LazyFrame, got DataFrame. Set `is_lazy=False` if you "
                "function doesn't doesn't need to use `.collect`, or make your frame "
                "before passing it to this function."
            )
            raise TypeError(msg)
        if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
            from narwhals.polars import DataFrame
            from narwhals.polars import Namespace

            return FrameTranslation(
                DataFrame(df, is_eager=is_eager, is_lazy=is_lazy), Namespace()
            )

    if (pd := get_pandas()) is not None and isinstance(df, pd.DataFrame):
        from narwhals.pandas_like.translate import translate_frame

        return FrameTranslation(
            *translate_frame(
                df,
                implementation="pandas",
                is_eager=is_eager,
                is_lazy=is_lazy,
            )
        )

    if (cudf := get_cudf()) is not None and isinstance(df, cudf.DataFrame):
        from narwhals.pandas_like.translate import translate_frame

        return FrameTranslation(
            *translate_frame(
                df, implementation="cudf", is_eager=is_eager, is_lazy=is_lazy
            )
        )

    if (mpd := get_modin()) is not None and isinstance(df, mpd.DataFrame):
        from narwhals.pandas_like.translate import translate_frame

        return FrameTranslation(
            *translate_frame(
                df, implementation="modin", is_eager=is_eager, is_lazy=is_lazy
            )
        )

    msg = f"Could not translate DataFrame {type(df)}, please open a feature request."
    raise NotImplementedError(msg)


def translate_series(
    series: Any,
) -> tuple[Series, Namespace]:
    if hasattr(series, "__narwhals_series__"):
        return SeriesTranslation(*series.__narwhals_series__())

    if (pl := get_polars()) is not None and isinstance(series, pl.Series):
        from narwhals.polars import Namespace
        from narwhals.polars import Series

        return SeriesTranslation(Series(series), Namespace())

    if (pd := get_pandas()) is not None and isinstance(series, pd.Series):
        from narwhals.pandas_like.translate import translate_series

        return SeriesTranslation(
            *translate_series(
                series,
                implementation="pandas",
            )
        )

    if (cudf := get_cudf()) is not None and isinstance(series, cudf.Series):
        from narwhals.pandas_like.translate import translate_series

        return SeriesTranslation(*translate_series(series, implementation="cudf"))

    if (mpd := get_modin()) is not None and isinstance(series, mpd.Series):
        from narwhals.pandas_like.translate import translate_series

        return SeriesTranslation(*translate_series(series, implementation="modin"))

    msg = f"Could not translate {type(series)}, please open a feature request."
    raise NotImplementedError(msg)


def translate_any(obj: Any) -> tuple[Series | DataFrame, Namespace]:
    try:
        return translate_series(obj)
    except NotImplementedError:
        return translate_frame(obj, is_eager=True)
