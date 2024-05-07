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
    from narwhals.dataframe import DataFrame
    from narwhals.dataframe import LazyFrame
    from narwhals.series import Series


def to_native(
    narwhals_object: LazyFrame | DataFrame | Series, *, strict: bool = True
) -> Any:
    """
    Convert Narwhals object to native one.

    Arguments:
        narwhals_object: Narwhals object.
        strict: whether to raise on non-Narwhals input.

    Returns:
        Object of class that user started with.
    """
    from narwhals.dataframe import BaseFrame
    from narwhals.series import Series

    if isinstance(narwhals_object, BaseFrame):
        return (
            narwhals_object._dataframe
            if narwhals_object._is_polars
            else narwhals_object._dataframe._dataframe
        )
    if isinstance(narwhals_object, Series):
        return (
            narwhals_object._series
            if narwhals_object._is_polars
            else narwhals_object._series._series
        )

    if strict:  # pragma: no cover (todo)
        msg = (
            f"Expected Narwhals object, got {type(narwhals_object)}."  # pragma: no cover
        )
        raise TypeError(msg)  # pragma: no cover
    return narwhals_object  # pragma: no cover (todo)


@overload
def from_native(
    native_dataframe: Any,
    *,
    strict: bool = ...,
    eager_only: Literal[True],
    series_only: None = ...,
    allow_series: Literal[True],
) -> DataFrame | Series: ...


@overload
def from_native(
    native_dataframe: Any,
    *,
    strict: bool = ...,
    eager_only: None = ...,
    series_only: None = ...,
    allow_series: Literal[True],
) -> DataFrame | LazyFrame | Series: ...


@overload
def from_native(
    native_dataframe: Any,
    *,
    strict: bool = ...,
    eager_only: None = ...,
    series_only: Literal[True],
    allow_series: None = ...,
) -> Series: ...


@overload
def from_native(
    native_dataframe: Any,
    *,
    strict: bool = ...,
    eager_only: Literal[True],
    series_only: None = ...,
    allow_series: None = ...,
) -> DataFrame: ...


@overload
def from_native(
    native_dataframe: Any,
    *,
    strict: bool = ...,
    eager_only: None = ...,
    series_only: None = ...,
    allow_series: None = ...,
) -> DataFrame | LazyFrame: ...


def from_native(
    native_dataframe: Any,
    *,
    strict: bool = True,
    eager_only: bool | None = None,
    series_only: bool | None = None,
    allow_series: bool | None = None,
) -> DataFrame | LazyFrame | Series:
    """
    Convert dataframe to Narwhals DataFrame, LazyFrame, or Series.

    Arguments:
        native_dataframe: Raw dataframe from user.
            Depending on the other arguments, input object can be:

            - pandas.DataFrame
            - polars.DataFrame
            - polars.LazyFrame
            - anything with a `__narwhals_dataframe__` or `__narwhals_lazyframe__` method
            - pandas.Series
            - polars.Series
            - anything with a `__narwhals_series__` method
        strict: Whether to raise if object can't be converted (default) or
            to just leave it as-is.
        eager_only: Whether to only allow eager objects.
        series_only: Whether to only allow series.
        allow_series: Whether to allow series (default is only dataframe / lazyframe).

    Returns:
        narwhals.DataFrame or narwhals.LazyFrame or narwhals.Series
    """
    from narwhals.dataframe import DataFrame
    from narwhals.dataframe import LazyFrame
    from narwhals.series import Series

    if series_only:
        allow_series = True
    # todo: raise on invalid combinations

    if (pl := get_polars()) is not None and isinstance(native_dataframe, pl.DataFrame):
        if series_only:  # pragma: no cover (todo)
            raise TypeError("Cannot only use `series_only` with polars.DataFrame")
        return DataFrame(native_dataframe)
    elif (pl := get_polars()) is not None and isinstance(native_dataframe, pl.LazyFrame):
        if series_only:  # pragma: no cover (todo)
            raise TypeError("Cannot only use `series_only` with polars.LazyFrame")
        if eager_only:  # pragma: no cover (todo)
            raise TypeError("Cannot only use `eager_only` with polars.LazyFrame")
        return LazyFrame(native_dataframe)
    elif (
        (pd := get_pandas()) is not None
        and isinstance(native_dataframe, pd.DataFrame)
        or (mpd := get_modin()) is not None
        and isinstance(native_dataframe, mpd.DataFrame)
        or (cudf := get_cudf()) is not None
        and isinstance(native_dataframe, cudf.DataFrame)
    ):
        if series_only:  # pragma: no cover (todo)
            raise TypeError("Cannot only use `series_only` with dataframe")
        return DataFrame(native_dataframe)
    elif hasattr(native_dataframe, "__narwhals_dataframe__"):  # pragma: no cover
        if series_only:  # pragma: no cover (todo)
            raise TypeError("Cannot only use `series_only` with dataframe")
        return DataFrame(native_dataframe.__narwhals_dataframe__())
    elif hasattr(native_dataframe, "__narwhals_lazyframe__"):  # pragma: no cover
        if series_only:  # pragma: no cover (todo)
            raise TypeError("Cannot only use `series_only` with lazyframe")
        if eager_only:  # pragma: no cover (todo)
            raise TypeError("Cannot only use `eager_only` with lazyframe")
        return LazyFrame(native_dataframe.__narwhals_lazyframe__())
    elif (
        (pl := get_polars()) is not None
        and isinstance(native_dataframe, pl.Series)
        or (pl := get_polars()) is not None
        and isinstance(native_dataframe, pl.Series)
        or (
            (pd := get_pandas()) is not None
            and isinstance(native_dataframe, pd.Series)
            or (mpd := get_modin()) is not None
            and isinstance(native_dataframe, mpd.Series)
            or (cudf := get_cudf()) is not None
            and isinstance(native_dataframe, cudf.Series)
        )
    ):
        if not allow_series:  # pragma: no cover (todo)
            raise TypeError("Please set `allow_series=True`")
        return Series(native_dataframe)
    elif hasattr(native_dataframe, "__narwhals_series__"):  # pragma: no cover
        if not allow_series:  # pragma: no cover (todo)
            raise TypeError("Please set `allow_series=True`")
        return Series(native_dataframe.__narwhals_series__())
    elif strict:  # pragma: no cover
        msg = f"Expected pandas-like dataframe, Polars dataframe, or Polars lazyframe, got: {type(native_dataframe)}"
        raise TypeError(msg)
    return native_dataframe  # type: ignore[no-any-return]  # pragma: no cover (todo)


__all__ = [
    "get_pandas",
    "get_polars",
    "get_modin",
    "get_cudf",
    "to_native",
]
