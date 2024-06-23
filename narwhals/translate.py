from __future__ import annotations

import inspect
from functools import wraps
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import TypeVar
from typing import overload

from narwhals import dtypes
from narwhals.dataframe import DataFrame
from narwhals.dataframe import LazyFrame
from narwhals.dependencies import get_cudf
from narwhals.dependencies import get_modin
from narwhals.dependencies import get_pandas
from narwhals.dependencies import get_polars
from narwhals.expression import Expr
from narwhals.expression import all
from narwhals.expression import col
from narwhals.expression import len
from narwhals.expression import lit
from narwhals.expression import max
from narwhals.expression import mean
from narwhals.expression import min
from narwhals.expression import sum
from narwhals.functions import concat
from narwhals.series import Series
from narwhals.utils import maybe_align_index
from narwhals.utils import maybe_convert_dtypes
from narwhals.utils import maybe_set_index

if TYPE_CHECKING:
    from narwhals.dataframe import DataFrame
    from narwhals.dataframe import LazyFrame
    from narwhals.dtypes import DType
    from narwhals.expression import Expr
    from narwhals.series import Series

T = TypeVar("T")


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
    api_version: str | None = ...,
) -> DataFrame | Series: ...


@overload
def from_native(
    native_dataframe: Any,
    *,
    strict: bool = ...,
    eager_only: Literal[True],
    series_only: None = ...,
    allow_series: None = ...,
    api_version: str | None = ...,
) -> DataFrame: ...


@overload
def from_native(
    native_dataframe: Any,
    *,
    strict: bool = ...,
    eager_only: None = ...,
    series_only: None = ...,
    allow_series: Literal[True],
    api_version: str | None = ...,
) -> DataFrame | LazyFrame | Series: ...


@overload
def from_native(
    native_dataframe: Any,
    *,
    strict: bool = ...,
    eager_only: None = ...,
    series_only: Literal[True],
    allow_series: None = ...,
    api_version: str | None = ...,
) -> Series: ...


@overload
def from_native(
    native_dataframe: Any,
    *,
    strict: bool = ...,
    eager_only: None = ...,
    series_only: None = ...,
    allow_series: None = ...,
    api_version: str | None = ...,
) -> DataFrame | LazyFrame: ...


@overload
def from_native(
    native_dataframe: Any,
    *,
    strict: bool = ...,
    eager_only: bool | None = ...,
    series_only: bool | None = ...,
    allow_series: bool | None = ...,
    api_version: str | None = ...,
) -> DataFrame | LazyFrame | Series: ...


def from_native(  # noqa: PLR0913
    native_dataframe: Any,
    *,
    strict: bool = True,
    eager_only: bool | None = None,
    series_only: bool | None = None,
    allow_series: bool | None = None,
    api_version: str | None = None,
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
        return DataFrame(native_dataframe, api_version=api_version or "0.20")
    elif (pl := get_polars()) is not None and isinstance(native_dataframe, pl.LazyFrame):
        if series_only:  # pragma: no cover (todo)
            raise TypeError("Cannot only use `series_only` with polars.LazyFrame")
        if eager_only:  # pragma: no cover (todo)
            raise TypeError("Cannot only use `eager_only` with polars.LazyFrame")
        return LazyFrame(native_dataframe, api_version=api_version or "0.20")
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
        return DataFrame(native_dataframe, api_version=api_version or "0.20")
    elif hasattr(native_dataframe, "__narwhals_dataframe__"):  # pragma: no cover
        if series_only:  # pragma: no cover (todo)
            raise TypeError("Cannot only use `series_only` with dataframe")
        return DataFrame(
            native_dataframe.__narwhals_dataframe__(), api_version=api_version or "0.20"
        )
    elif hasattr(native_dataframe, "__narwhals_lazyframe__"):  # pragma: no cover
        if series_only:  # pragma: no cover (todo)
            raise TypeError("Cannot only use `series_only` with lazyframe")
        if eager_only:  # pragma: no cover (todo)
            raise TypeError("Cannot only use `eager_only` with lazyframe")
        return LazyFrame(
            native_dataframe.__narwhals_lazyframe__(), api_version=api_version or "0.20"
        )
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
        return Series(native_dataframe, api_version=api_version or "0.20")
    elif hasattr(native_dataframe, "__narwhals_series__"):  # pragma: no cover
        if not allow_series:  # pragma: no cover (todo)
            raise TypeError("Please set `allow_series=True`")
        return Series(
            native_dataframe.__narwhals_series__(), api_version=api_version or "0.20"
        )
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
            df = df.group_by("a").agg(nw.col("b").sum())
            return nw.to_native(df)
    ```

    you can just write

    ```python
    import narwhals as nw


    class Foo:
        @nw.narwhalify_method
        def func(self, df):
            return df.group_by("a").agg(nw.col("b").sum())
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


class StableAPI:
    api_version: str

    DataFrame = DataFrame
    LazyFrame = LazyFrame
    Series = Series
    Expr = Expr

    def __init__(self, api_version: str) -> None:
        self.api_version = api_version

    def all(self, *columns: str | Iterable[str], api_version: str | None = None) -> Expr:
        return all(*columns, api_version=self.api_version)

    def col(self, *names: str | Iterable[str], api_version: str | None = None) -> Expr:
        return col(*names, api_version=self.api_version)

    concat = concat

    def from_native(  # noqa: PLR0913
        self,
        native_dataframe: Any,
        *,
        strict: bool = True,
        eager_only: bool | None = None,
        series_only: bool | None = None,
        allow_series: bool | None = None,
    ) -> DataFrame | LazyFrame | Series:
        return from_native(
            native_dataframe,
            strict=strict,
            eager_only=eager_only,
            series_only=series_only,
            allow_series=allow_series,
            api_version=self.api_version,
        )

    def get_native_namespace(self, obj: Any) -> Any:
        return get_native_namespace(obj)

    def len(self, *, api_version: str | None = None) -> Expr:
        return len(api_version=self.api_version)

    def maybe_align_index(self, lhs: T, rhs: Series | LazyFrame | DataFrame) -> T:
        return maybe_align_index(lhs, rhs)

    def maybe_set_index(self, df: T, column_names: str | list[str]) -> T:
        return maybe_set_index(df, column_names)

    def maybe_convert_dtypes(self, df: T, *args: bool, **kwargs: bool | str) -> T:
        return maybe_convert_dtypes(df, *args, **kwargs)

    def lit(
        self, value: Any, dtype: DType | None = None, api_version: str | None = None
    ) -> Expr:
        return lit(value, dtype, api_version=api_version)

    def max(self, *columns: str, api_version: str | None = None) -> Expr:
        return max(*columns, api_version=api_version)

    def mean(self, *columns: str, api_version: str | None = None) -> Expr:
        return mean(*columns, api_version=api_version)

    def min(self, *columns: str, api_version: str | None = None) -> Expr:
        return min(*columns, api_version=api_version)

    def to_native(
        self, narwhals_object: LazyFrame | DataFrame | Series, *, strict: bool = True
    ) -> Any:
        return to_native(narwhals_object, strict=strict)

    def sum(self, *columns: str | Iterable[str], api_version: str | None = None) -> Expr:
        return sum(*columns, api_version=self.api_version)

    Boolean = dtypes.Boolean
    Int64 = dtypes.Int64
    Int32 = dtypes.Int32
    Int16 = dtypes.Int16
    Int8 = dtypes.Int8
    UInt64 = dtypes.UInt64
    UInt32 = dtypes.UInt32
    UInt16 = dtypes.UInt16
    UInt8 = dtypes.UInt8
    Float64 = dtypes.Float64
    Float32 = dtypes.Float32
    String = dtypes.String
    Categorical = dtypes.Categorical


__all__ = [
    "get_pandas",
    "get_polars",
    "get_modin",
    "get_cudf",
    "get_native_namespace",
    "to_native",
    "narwhalify",
]
