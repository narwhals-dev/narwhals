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

from narwhals import dependencies
from narwhals import dtypes
from narwhals import selectors
from narwhals import typing
from narwhals import utils
from narwhals import versions
from narwhals._dataframe import DataFrame
from narwhals._dataframe import LazyFrame
from narwhals._expression import Expr
from narwhals._expression import all
from narwhals._expression import col
from narwhals._expression import len
from narwhals._expression import lit
from narwhals._expression import max
from narwhals._expression import mean
from narwhals._expression import min
from narwhals._expression import sum
from narwhals._expression import sum_horizontal
from narwhals._functions import concat
from narwhals._functions import show_versions
from narwhals._series import Series
from narwhals.dependencies import get_cudf
from narwhals.dependencies import get_modin
from narwhals.dependencies import get_pandas
from narwhals.dependencies import get_polars
from narwhals.utils import maybe_align_index
from narwhals.utils import maybe_convert_dtypes
from narwhals.utils import maybe_set_index
from narwhals.versions import validate_api_version

if TYPE_CHECKING:
    from narwhals.dtypes import DType
    from narwhals.typing import API_VERSION
    from narwhals.typing import IntoExpr

T = TypeVar("T")

DEFAULT_API_VERSION: API_VERSION = "0.20"

if TYPE_CHECKING:
    import narwhals as nw


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
    from narwhals._dataframe import BaseFrame
    from narwhals._series import Series

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
    api_version: API_VERSION = ...,
) -> DataFrame | Series: ...


@overload
def from_native(
    native_dataframe: Any,
    *,
    strict: bool = ...,
    eager_only: Literal[True],
    series_only: None = ...,
    allow_series: None = ...,
    api_version: API_VERSION = ...,
) -> DataFrame: ...


@overload
def from_native(
    native_dataframe: Any,
    *,
    strict: bool = ...,
    eager_only: None = ...,
    series_only: None = ...,
    allow_series: Literal[True],
    api_version: API_VERSION = ...,
) -> DataFrame | LazyFrame | Series: ...


@overload
def from_native(
    native_dataframe: Any,
    *,
    strict: bool = ...,
    eager_only: None = ...,
    series_only: Literal[True],
    allow_series: None = ...,
    api_version: API_VERSION = ...,
) -> Series: ...


@overload
def from_native(
    native_dataframe: Any,
    *,
    strict: bool = ...,
    eager_only: None = ...,
    series_only: None = ...,
    allow_series: None = ...,
    api_version: API_VERSION = ...,
) -> DataFrame | LazyFrame: ...


@overload
def from_native(
    native_dataframe: Any,
    *,
    strict: bool = ...,
    eager_only: bool | None = ...,
    series_only: bool | None = ...,
    allow_series: bool | None = ...,
    api_version: API_VERSION = ...,
) -> DataFrame | LazyFrame | Series: ...


def from_native(
    native_dataframe: Any,
    *,
    strict: bool = True,
    eager_only: bool | None = None,
    series_only: bool | None = None,
    allow_series: bool | None = None,
    api_version: API_VERSION = DEFAULT_API_VERSION,
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
        api_version: Narwhals API version to use, if you want to ensure perfect
            backwards-compatibility. The easiest way to use this is to set it once
            in `narwhals.StableAPI` and then just import that.

    Returns:
        narwhals.DataFrame or narwhals.LazyFrame or narwhals.Series
    """
    from narwhals._dataframe import DataFrame
    from narwhals._dataframe import LazyFrame
    from narwhals._series import Series

    if series_only:
        allow_series = True
    # todo: raise on invalid combinations

    if (pl := get_polars()) is not None and isinstance(native_dataframe, pl.DataFrame):
        if series_only:  # pragma: no cover (todo)
            raise TypeError("Cannot only use `series_only` with polars.DataFrame")
        return DataFrame(native_dataframe, api_version=api_version or DEFAULT_API_VERSION)
    elif (pl := get_polars()) is not None and isinstance(native_dataframe, pl.LazyFrame):
        if series_only:  # pragma: no cover (todo)
            raise TypeError("Cannot only use `series_only` with polars.LazyFrame")
        if eager_only:  # pragma: no cover (todo)
            raise TypeError("Cannot only use `eager_only` with polars.LazyFrame")
        return LazyFrame(native_dataframe, api_version=api_version or DEFAULT_API_VERSION)
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
        return DataFrame(native_dataframe, api_version=api_version or DEFAULT_API_VERSION)
    elif hasattr(native_dataframe, "__narwhals_dataframe__"):  # pragma: no cover
        if series_only:  # pragma: no cover (todo)
            raise TypeError("Cannot only use `series_only` with dataframe")
        return DataFrame(
            native_dataframe.__narwhals_dataframe__(),
            api_version=api_version or DEFAULT_API_VERSION,
        )
    elif hasattr(native_dataframe, "__narwhals_lazyframe__"):  # pragma: no cover
        if series_only:  # pragma: no cover (todo)
            raise TypeError("Cannot only use `series_only` with lazyframe")
        if eager_only:  # pragma: no cover (todo)
            raise TypeError("Cannot only use `eager_only` with lazyframe")
        return LazyFrame(
            native_dataframe.__narwhals_lazyframe__(),
            api_version=api_version or DEFAULT_API_VERSION,
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
        return Series(native_dataframe, api_version=api_version or DEFAULT_API_VERSION)
    elif hasattr(native_dataframe, "__narwhals_series__"):  # pragma: no cover
        if not allow_series:  # pragma: no cover (todo)
            raise TypeError("Please set `allow_series=True`")
        return Series(
            native_dataframe.__narwhals_series__(),
            api_version=api_version or DEFAULT_API_VERSION,
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
    api_version: API_VERSION = DEFAULT_API_VERSION,
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
                api_version=api_version,
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
    api_version: API_VERSION = DEFAULT_API_VERSION,
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
                api_version=api_version,
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
    """
    Instantiate a perfectly-backwards-compatible Narwhals namespace.

    This concept is inspired by
    [Rust editions](https://doc.rust-lang.org/edition-guide/editions/).

    If you instantiate a `StableAPI` object, then Narhwals promises that your code
    will never unintentionally break, even as newer (possibly API-breaking) versions
    of pandas/Polars/etc come out. You may sometimes need to bump your minimum
    Narwhals version, but you should never need to worry about your code going
    out-of-date.

    Arguments:
        api_version: Polars API version to follow. Allowed values are:

        - 0.20
        - 1.0

    Examples:
        Suppose you maintain a library called `skbeer`. Here's an example
        of how you could use Narwhals in a perfectly-backwards compatible way.

        First, in some project file (say, `skbeer/_narwhals.py`), instantiate:

        >>> from narwhals import StableAPI
        >>> nw = StableAPI("0.20")

        Then, in other modules, if you'd like to use Narwhals, import it as follows
        (note: replace `skbeer` with the name of your module):

        >>> from skbeer._narwhals import nw  # doctest: +SKIP
        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.with_columns(nw.col("a").shift(1))

        Suppose hypothetically for the sake of argument that in version 1.0, Polars
        was to change the direction in which `shift` operates. This hasn't happened,
        but presumably, at some point, Narwhals will be hit by a backwards-incompatible
        API change in Polars.

        At that point, given that you instantiated `StableAPI` with `api_version='0.20'`,
        then your code will carry on operating as it always did, even with newer versions
        of Polars! All you'd need to do is to bump the minimum version of Narwhals to one
        where we handle the different Polars' API behaviours.

        We follow development of pandas and Polars _very_ closely, and so we expect to be
        able to make compatible Narwhals releases well in advance of pandas/Polars ones.
    """

    api_version: API_VERSION

    DataFrame = DataFrame
    LazyFrame = LazyFrame
    Series = Series
    Expr = Expr
    selectors = selectors
    dtypes = dtypes
    dependencies = dependencies
    utils = utils
    typing = typing
    versions = versions

    def __init__(self, api_version: API_VERSION) -> None:
        validate_api_version(api_version)
        self.api_version = api_version

    def all(self) -> nw.Expr:
        return all(api_version=self.api_version)

    def col(self, *names: str | Iterable[str]) -> nw.Expr:
        return col(*names, api_version=self.api_version)

    def concat(
        self,
        items: Iterable[nw.DataFrame | nw.LazyFrame],
        *,
        how: Literal["horizontal", "vertical"] = "vertical",
    ) -> nw.DataFrame | nw.LazyFrame:
        return concat(items, how=how)

    @overload
    def from_native(
        self,
        native_dataframe: Any,
        *,
        strict: bool = ...,
        eager_only: Literal[True],
        series_only: None = ...,
        allow_series: Literal[True],
    ) -> nw.DataFrame | nw.Series: ...

    @overload
    def from_native(
        self,
        native_dataframe: Any,
        *,
        strict: bool = ...,
        eager_only: Literal[True],
        series_only: None = ...,
        allow_series: None = ...,
    ) -> nw.DataFrame: ...

    @overload
    def from_native(
        self,
        native_dataframe: Any,
        *,
        strict: bool = ...,
        eager_only: None = ...,
        series_only: None = ...,
        allow_series: Literal[True],
    ) -> nw.DataFrame | nw.LazyFrame | nw.Series: ...

    @overload
    def from_native(
        self,
        native_dataframe: Any,
        *,
        strict: bool = ...,
        eager_only: None = ...,
        series_only: Literal[True],
        allow_series: None = ...,
    ) -> nw.Series: ...

    @overload
    def from_native(
        self,
        native_dataframe: Any,
        *,
        strict: bool = ...,
        eager_only: None = ...,
        series_only: None = ...,
        allow_series: None = ...,
    ) -> nw.DataFrame | nw.LazyFrame: ...

    @overload
    def from_native(
        self,
        native_dataframe: Any,
        *,
        strict: bool = ...,
        eager_only: bool | None = ...,
        series_only: bool | None = ...,
        allow_series: bool | None = ...,
    ) -> nw.DataFrame | nw.LazyFrame | nw.Series: ...

    def from_native(
        self,
        native_dataframe: Any,
        *,
        strict: bool = True,
        eager_only: bool | None = None,
        series_only: bool | None = None,
        allow_series: bool | None = None,
    ) -> nw.DataFrame | nw.LazyFrame | nw.Series:
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

    def len(self) -> nw.Expr:
        return len(api_version=self.api_version)

    def maybe_align_index(
        self, lhs: T, rhs: nw.Series | nw.LazyFrame | nw.DataFrame
    ) -> T:
        return maybe_align_index(lhs, rhs)

    def maybe_set_index(self, df: T, column_names: str | list[str]) -> T:
        return maybe_set_index(df, column_names)

    def maybe_convert_dtypes(self, df: T, *args: bool, **kwargs: bool | str) -> T:
        return maybe_convert_dtypes(df, *args, **kwargs)

    def lit(self, value: Any, dtype: DType | None = None) -> nw.Expr:
        return lit(value, dtype, api_version=self.api_version)

    def max(self, *columns: str) -> nw.Expr:
        return max(*columns, api_version=self.api_version)

    def mean(self, *columns: str) -> nw.Expr:
        return mean(*columns, api_version=self.api_version)

    def min(self, *columns: str) -> nw.Expr:
        return min(*columns, api_version=self.api_version)

    def narwhalify(
        self,
        func: Callable[..., Any] | None = None,
        *,
        strict: bool = True,
        eager_only: bool | None = None,
        series_only: bool | None = None,
        allow_series: bool | None = None,
    ) -> Callable[..., Any]:
        return narwhalify(
            func,
            strict=strict,
            eager_only=eager_only,
            series_only=series_only,
            allow_series=allow_series,
            api_version=self.api_version,
        )

    def narwhalify_method(
        self,
        func: Callable[..., Any] | None = None,
        *,
        strict: bool = True,
        eager_only: bool | None = None,
        series_only: bool | None = None,
        allow_series: bool | None = None,
    ) -> Callable[..., Any]:
        return narwhalify_method(
            func,
            strict=strict,
            eager_only=eager_only,
            series_only=series_only,
            allow_series=allow_series,
            api_version=self.api_version,
        )

    def sum(self, *columns: str | Iterable[str]) -> nw.Expr:
        return sum(*columns, api_version=self.api_version)

    def sum_horizontal(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> nw.Expr:
        return sum_horizontal(*exprs, api_version=self.api_version)

    def show_versions(self) -> None:
        return show_versions()

    def to_native(
        self,
        narwhals_object: nw.LazyFrame | nw.DataFrame | nw.Series,
        *,
        strict: bool = True,
    ) -> Any:
        return to_native(narwhals_object, strict=strict)

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
    Datetime = dtypes.Datetime


__all__ = [
    "get_pandas",
    "get_polars",
    "get_modin",
    "get_cudf",
    "get_native_namespace",
    "to_native",
    "narwhalify",
]
