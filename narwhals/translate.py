from __future__ import annotations

import inspect
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


def get_native_namespace(obj: Any) -> Any:
    """
    Get native namespace from object.

    Examples:
        >>> import polars as pl
        >>> import pandas as pd
        >>> import narwhals as nw
        >>> df = nw.from_native(pd.DataFrame({'a': [1,2,3]}))
        >>> nw.get_native_namespace(df)
        <module 'pandas'...>
        >>> df = nw.from_native(pl.DataFrame({'a': [1,2,3]}))
        >>> nw.get_native_namespace(df)
        <module 'polars'...>
    """
    return obj.__native_namespace__()


def narwhalify(
    func: Callable[..., Any] | None = None,
    *,
    strict: bool = True,
    eager_only: bool | None = None,
    series_only: bool | None = None,
    allow_series: bool | None = None,
) -> Callable[..., Any]:
    """
    Decorate function so it becomes dataframe-agnostic.

    Instead of writing

    ```python
    import narwhals as nw

    def func(df_any):
        df = nw.from_native(df_any, strict=False)
        df = df.group_by('a').agg(nw.col('b').sum())
        return nw.to_native(df)
    ```

    you can just write

    ```python
    import narwhals as nw

    @nw.narwhalify
    def func(df):
        return df.group_by('a').agg(nw.col('b').sum())
    ```

    You can also pass in extra arguments, e.g.

    ```python
    @nw.narhwalify(eager_only=True)
    ```

    that will get passed down to `nw.from_native`.

    Arguments:
        func: Function to wrap in a `from_native`-`to_native` block.
        strict: Whether to raise if object can't be converted (default) or
            to just leave it as-is.
        eager_only: Whether to only allow eager objects.
        series_only: Whether to only allow series.
        allow_series: Whether to allow series (default is only dataframe / lazyframe).

    See Also:
        narwhalify_method: If you want to narwhalify a class method, use that instead.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if args:
                df_any = args[0]
            elif kwargs:
                params = list(inspect.signature(func).parameters.keys())
                first_key = params[0]
                df_any = kwargs[first_key]
            else:
                raise TypeError("Expected function which takes at least one argument.")
            df = from_native(
                df_any,
                strict=strict,
                eager_only=eager_only,
                series_only=series_only,
                allow_series=allow_series,
            )
            if args:
                result = func(df, *args[1:], **kwargs)
            else:
                kwargs[first_key] = df
                result = func(**kwargs)
            return to_native(result, strict=strict)

        return wrapper

    # If func is None, it means the decorator is used with arguments
    if func is None:
        return decorator
    else:
        # If func is not None, it means the decorator is used without arguments
        return decorator(func)


def narwhalify_method(
    func: Callable[..., Any] | None = None,
    *,
    strict: bool = True,
    eager_only: bool | None = None,
    series_only: bool | None = None,
    allow_series: bool | None = None,
) -> Callable[..., Any]:
    """
    Decorate method so it becomes dataframe-agnostic.

    Instead of writing

    ```python
    import narwhals as nw

    class Foo:
        def func(self, df_any):
            df = nw.from_native(df_any, strict=False)
            df = df.group_by('a').agg(nw.col('b').sum())
            return nw.to_native(df)
    ```

    you can just write

    ```python
    import narwhals as nw

    class Foo:
        @nw.narwhalify_method
        def func(self, df):
            return df.group_by('a').agg(nw.col('b').sum())
    ```

    You can also pass in extra arguments, e.g.

    ```python
    @nw.narhwalify_method(eager_only=True)
    ```

    that will get passed down to `nw.from_native`.

    Arguments:
        func: Function to wrap in a `from_native`-`to_native` block.
        strict: Whether to raise if object can't be converted (default) or
            to just leave it as-is.
        eager_only: Whether to only allow eager objects.
        series_only: Whether to only allow series.
        allow_series: Whether to allow series (default is only dataframe / lazyframe).
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if args:
                df_any = args[0]
            elif kwargs:
                params = list(inspect.signature(func).parameters.keys())
                if params[0] not in ("cls", "self"):
                    msg = (
                        "`@nw.narwhalify_method` is meant to be called on class methods, "
                        "where the first argument is typically `cls` or `self` - however, yours "
                        f"is: {params[0]}."
                    )
                    raise TypeError(msg)
                first_key = params[1]
                df_any = kwargs[first_key]
            else:
                raise TypeError("Expected function which takes at least one argument.")
            df = from_native(
                df_any,
                strict=strict,
                eager_only=eager_only,
                series_only=series_only,
                allow_series=allow_series,
            )
            if args:
                result = func(self, df, *args[1:], **kwargs)
            else:
                kwargs[first_key] = df
                result = func(self, **kwargs)
            return to_native(result, strict=strict)

        return wrapper

    # If func is None, it means the decorator is used with arguments
    if func is None:
        return decorator
    else:
        # If func is not None, it means the decorator is used without arguments
        return decorator(func)


__all__ = [
    "get_pandas",
    "get_polars",
    "get_modin",
    "get_cudf",
    "get_native_namespace",
    "to_native",
    "narwhalify",
]
