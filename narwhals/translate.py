from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import overload

from narwhals.dependencies import get_cudf
from narwhals.dependencies import get_modin
from narwhals.dependencies import get_pandas
from narwhals.dependencies import get_polars
from narwhals.dependencies import get_pyarrow

if TYPE_CHECKING:
    from narwhals.dataframe import DataFrame
    from narwhals.dataframe import LazyFrame
    from narwhals.series import Series


def to_native(narwhals_object: Any, *, strict: bool = True) -> Any:
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
    eager_only: None = ...,
    series_only: None = ...,
    allow_series: None = ...,
) -> DataFrame | LazyFrame: ...


@overload
def from_native(
    native_dataframe: Any,
    *,
    strict: bool,
    eager_only: bool | None,
    series_only: bool | None,
    allow_series: bool | None,
) -> DataFrame | LazyFrame | Series: ...


def from_native(  # noqa: PLR0911 - This will become easier once #363 is merged
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
    # TODO: raise on invalid combinations

    if (pl := get_polars()) is not None and isinstance(native_dataframe, pl.DataFrame):
        if series_only:  # pragma: no cover (todo)
            error_message = "Cannot only use `series_only` with polars.DataFrame"
            raise TypeError(error_message)
        return DataFrame(native_dataframe)
    elif (pl := get_polars()) is not None and isinstance(native_dataframe, pl.LazyFrame):
        if series_only:  # pragma: no cover (todo)
            error_message = "Cannot only use `series_only` with polars.LazyFrame"
            raise TypeError(error_message)
        if eager_only:  # pragma: no cover (todo)
            error_message = "Cannot only use `eager_only` with polars.LazyFrame"
            raise TypeError(error_message)
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
            error_message = "Cannot only use `series_only` with dataframe"
            raise TypeError(error_message)
        return DataFrame(native_dataframe)
    elif (pa := get_pyarrow()) is not None and isinstance(native_dataframe, pa.Table):
        if series_only:  # pragma: no cover (todo)
            error_message = "Cannot only use `series_only` with arrow table"
            raise TypeError(error_message)
        return DataFrame(native_dataframe)
    elif hasattr(native_dataframe, "__narwhals_dataframe__"):  # pragma: no cover
        if series_only:  # pragma: no cover (todo)
            error_message = "Cannot only use `series_only` with dataframe"
            raise TypeError(error_message)
        return DataFrame(native_dataframe.__narwhals_dataframe__())
    elif hasattr(native_dataframe, "__narwhals_lazyframe__"):  # pragma: no cover
        if series_only:  # pragma: no cover (todo)
            error_message = "Cannot only use `series_only` with lazyframe"
            raise TypeError(error_message)
        if eager_only:  # pragma: no cover (todo)
            error_message = "Cannot only use `eager_only` with lazyframe"
            raise TypeError(error_message)
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
            or (pa := get_pyarrow()) is not None
            and isinstance(native_dataframe, pa.ChunkedArray)
        )
    ):
        if not allow_series:  # pragma: no cover (todo)
            error_messagge = "Please set `allow_series=True`"
            raise TypeError(error_messagge)
        return Series(native_dataframe)
    elif hasattr(native_dataframe, "__narwhals_series__"):  # pragma: no cover
        if not allow_series:  # pragma: no cover (todo)
            error_messagge = "Please set `allow_series=True`"
            raise TypeError(error_messagge)
        return Series(native_dataframe.__narwhals_series__())
    elif strict:  # pragma: no cover
        msg = f"Expected pandas-like dataframe, Polars dataframe, or Polars lazyframe, got: {type(native_dataframe)}"
        raise TypeError(msg)
    return native_dataframe  # type: ignore[no-any-return]  # pragma: no cover (todo)


def get_native_namespace(obj: Any) -> Any:
    """
    Get native namespace from object.

    Examples:
        >>> import polars as pl
        >>> import pandas as pd
        >>> import narwhals as nw
        >>> df = nw.from_native(pd.DataFrame({"a": [1, 2, 3]}))
        >>> nw.get_native_namespace(df)
        <module 'pandas'...>
        >>> df = nw.from_native(pl.DataFrame({"a": [1, 2, 3]}))
        >>> nw.get_native_namespace(df)
        <module 'polars'...>
    """
    return obj.__native_namespace__()


def narwhalify(
    func: Callable[..., Any] | None = None,
    *,
    strict: bool = False,
    eager_only: bool | None = False,
    series_only: bool | None = False,
    allow_series: bool | None = True,
) -> Callable[..., Any]:
    """
    Decorate function so it becomes dataframe-agnostic.

    `narwhalify` will try to convert any dataframe/series-like object into the narwhal
    respective DataFrame/Series, while leaving the other parameters as they are.

    Similarly, if the output of the function is a narwhals DataFrame or Series, it will be
    converted back to the original dataframe/series type, while if the output is another
    type it will be left as is.

    By setting `strict=True`, then every input and every output will be required to be a
    dataframe/series-like object.

    Instead of writing

    ```python
    import narwhals as nw


    def func(df_any):
        df = nw.from_native(df_any, strict=False)
        df = df.group_by("a").agg(nw.col("b").sum())
        return nw.to_native(df)
    ```

    you can just write

    ```python
    import narwhals as nw


    @nw.narwhalify
    def func(df):
        return df.group_by("a").agg(nw.col("b").sum())
    ```

    You can also pass in extra arguments, e.g.

    ```python
    @nw.narhwalify(eager_only=True)
    ```

    that will get passed down to `nw.from_native`.

    Arguments:
        func: Function to wrap in a `from_native`-`to_native` block.
        strict: Whether to raise if object can't be converted or to just leave it as-is
            (default).
        eager_only: Whether to only allow eager objects.
        series_only: Whether to only allow series.
        allow_series: Whether to allow series (default is only dataframe / lazyframe).
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            args = [
                from_native(
                    arg,
                    strict=strict,
                    eager_only=eager_only,
                    series_only=series_only,
                    allow_series=allow_series,
                )
                for arg in args
            ]  # type: ignore[assignment]

            kwargs = {
                name: from_native(
                    value,
                    strict=strict,
                    eager_only=eager_only,
                    series_only=series_only,
                    allow_series=allow_series,
                )
                for name, value in kwargs.items()
            }

            backends = {
                b()
                for v in [*args, *kwargs.values()]
                if (b := getattr(v, "__native_namespace__", None))
            }

            if len(backends) > 1:
                msg = "Found multiple backends. Make sure that all dataframe/series inputs come from the same backend."
                raise ValueError(msg)

            result = func(*args, **kwargs)

            return to_native(result, strict=strict)

        return wrapper

    if func is None:
        return decorator
    else:
        # If func is not None, it means the decorator is used without arguments
        return decorator(func)


__all__ = [
    "get_native_namespace",
    "to_native",
    "narwhalify",
]
