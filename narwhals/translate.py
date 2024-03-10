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
) -> DataFrame:
    if is_eager and is_lazy:
        msg = "Only one of `is_eager` and `is_lazy` can be True."
        raise ValueError(msg)

    if hasattr(df, "__narwhals_frame__"):
        return df.__narwhals_frame__(is_eager=is_eager, is_lazy=is_lazy)  # type: ignore[no-any-return]

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
            from narwhals.polars import PolarsDataFrame

            return PolarsDataFrame(df, is_eager=is_eager, is_lazy=is_lazy)

    if (pd := get_pandas()) is not None and isinstance(df, pd.DataFrame):
        from narwhals.pandas_like.translate import translate_frame

        return translate_frame(
            df,
            implementation="pandas",
            is_eager=is_eager,
            is_lazy=is_lazy,
        )

    if (cudf := get_cudf()) is not None and isinstance(df, cudf.DataFrame):
        from narwhals.pandas_like.translate import translate_frame

        return translate_frame(
            df, implementation="cudf", is_eager=is_eager, is_lazy=is_lazy
        )

    if (mpd := get_modin()) is not None and isinstance(df, mpd.DataFrame):
        from narwhals.pandas_like.translate import translate_frame

        return translate_frame(
            df, implementation="modin", is_eager=is_eager, is_lazy=is_lazy
        )

    msg = f"Could not translate DataFrame {type(df)}, please open a feature request."
    raise NotImplementedError(msg)


def translate_series(
    series: Any,
) -> Series:
    if hasattr(series, "__narwhals_series__"):
        return series.__narwhals_series__()  # type: ignore[no-any-return]

    if (pl := get_polars()) is not None and isinstance(series, pl.Series):
        from narwhals.polars import PolarsSeries

        return PolarsSeries(series)

    if (pd := get_pandas()) is not None and isinstance(series, pd.Series):
        from narwhals.pandas_like.translate import translate_series

        return translate_series(
            series,
            implementation="pandas",
        )

    if (cudf := get_cudf()) is not None and isinstance(series, cudf.Series):
        from narwhals.pandas_like.translate import translate_series

        return translate_series(series, implementation="cudf")

    if (mpd := get_modin()) is not None and isinstance(series, mpd.Series):
        from narwhals.pandas_like.translate import translate_series

        return translate_series(series, implementation="modin")

    msg = f"Could not translate {type(series)}, please open a feature request."
    raise NotImplementedError(msg)


def translate_any(obj: Any) -> Series | DataFrame:
    try:
        return translate_series(obj)
    except NotImplementedError:
        return translate_frame(obj, is_eager=True)


def get_namespace(obj: Any) -> Namespace:
    from narwhals.containers import is_cudf
    from narwhals.containers import is_modin
    from narwhals.containers import is_pandas
    from narwhals.containers import is_polars
    from narwhals.pandas_like.dataframe import PandasDataFrame
    from narwhals.pandas_like.series import PandasSeries
    from narwhals.polars import PolarsDataFrame
    from narwhals.polars import PolarsSeries

    if obj == "pandas":
        from narwhals.pandas_like.namespace import Namespace

        return Namespace(implementation="pandas")
    if obj == "polars":
        from narwhals.polars import Namespace  # type: ignore[assignment]

        return Namespace()  # type: ignore[call-arg]

    if isinstance(obj, (PandasDataFrame, PandasSeries)):
        from narwhals.pandas_like.namespace import Namespace

        return Namespace(implementation=obj._implementation)
    if isinstance(obj, (PolarsDataFrame, PolarsSeries)):
        from narwhals.polars import Namespace  # type: ignore[assignment]

        return Namespace()  # type: ignore[call-arg]

    if is_polars(obj):
        from narwhals.polars import Namespace  # type: ignore[assignment]

        return Namespace(implementation="polars")
    if is_pandas(obj):
        from narwhals.pandas_like.namespace import Namespace

        return Namespace(implementation="pandas")
    if is_modin(obj):
        from narwhals.pandas_like.namespace import Namespace

        return Namespace(implementation="modin")
    if is_cudf(obj):
        from narwhals.pandas_like.namespace import Namespace

        return Namespace(implementation="cudf")

    raise NotImplementedError


def to_native(obj: Any) -> Any:
    from narwhals.pandas_like.dataframe import PandasDataFrame
    from narwhals.pandas_like.series import PandasSeries
    from narwhals.polars import PolarsDataFrame
    from narwhals.polars import PolarsSeries

    if isinstance(obj, PandasDataFrame):
        return obj._dataframe
    if isinstance(obj, PandasSeries):
        return obj._series
    if isinstance(obj, PolarsDataFrame):
        return obj._dataframe
    if isinstance(obj, PolarsSeries):
        return obj._series

    msg = f"Expected Narwhals object, got {type(obj)}."
    raise TypeError(msg)


__all__ = [
    "translate_frame",
    "translate_series",
    "translate_any",
    "get_pandas",
    "get_polars",
    "get_namespace",
    "to_native",
]
