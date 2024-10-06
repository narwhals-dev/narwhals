from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import Sequence
from typing import TypeVar
from typing import overload

import narwhals as nw
from narwhals import dependencies
from narwhals import selectors
from narwhals.dataframe import DataFrame as NwDataFrame
from narwhals.dataframe import LazyFrame as NwLazyFrame
from narwhals.expr import Expr as NwExpr
from narwhals.expr import Then as NwThen
from narwhals.expr import When as NwWhen
from narwhals.expr import when as nw_when
from narwhals.functions import _from_dict_impl
from narwhals.functions import _new_series_impl
from narwhals.functions import show_versions
from narwhals.schema import Schema as NwSchema
from narwhals.series import Series as NwSeries
from narwhals.stable.v1.dtypes import Array
from narwhals.stable.v1.dtypes import Boolean
from narwhals.stable.v1.dtypes import Categorical
from narwhals.stable.v1.dtypes import Date
from narwhals.stable.v1.dtypes import Datetime
from narwhals.stable.v1.dtypes import Duration
from narwhals.stable.v1.dtypes import Enum
from narwhals.stable.v1.dtypes import Float32
from narwhals.stable.v1.dtypes import Float64
from narwhals.stable.v1.dtypes import Int8
from narwhals.stable.v1.dtypes import Int16
from narwhals.stable.v1.dtypes import Int32
from narwhals.stable.v1.dtypes import Int64
from narwhals.stable.v1.dtypes import List
from narwhals.stable.v1.dtypes import Object
from narwhals.stable.v1.dtypes import String
from narwhals.stable.v1.dtypes import Struct
from narwhals.stable.v1.dtypes import UInt8
from narwhals.stable.v1.dtypes import UInt16
from narwhals.stable.v1.dtypes import UInt32
from narwhals.stable.v1.dtypes import UInt64
from narwhals.stable.v1.dtypes import Unknown
from narwhals.translate import _from_native_impl
from narwhals.translate import get_native_namespace as nw_get_native_namespace
from narwhals.translate import to_native
from narwhals.typing import IntoDataFrameT
from narwhals.typing import IntoFrameT
from narwhals.utils import is_ordered_categorical as nw_is_ordered_categorical
from narwhals.utils import maybe_align_index as nw_maybe_align_index
from narwhals.utils import maybe_convert_dtypes as nw_maybe_convert_dtypes
from narwhals.utils import maybe_get_index as nw_maybe_get_index
from narwhals.utils import maybe_reset_index as nw_maybe_reset_index
from narwhals.utils import maybe_set_index as nw_maybe_set_index

if TYPE_CHECKING:
    from types import ModuleType

    from typing_extensions import Self

    from narwhals.dtypes import DType
    from narwhals.typing import IntoExpr

T = TypeVar("T")


class DataFrame(NwDataFrame[IntoDataFrameT]):
    """
    Narwhals DataFrame, backed by a native dataframe.

    The native dataframe might be pandas.DataFrame, polars.DataFrame, ...

    This class is not meant to be instantiated directly - instead, use
    `narwhals.from_native`.
    """

    # We need to override any method which don't return Self so that type
    # annotations are correct.

    @property
    def _series(self) -> type[Series]:
        return Series

    @property
    def _lazyframe(self) -> type[LazyFrame[Any]]:
        return LazyFrame

    @overload
    def __getitem__(self, item: tuple[Sequence[int], slice]) -> Self: ...
    @overload
    def __getitem__(self, item: tuple[Sequence[int], Sequence[int]]) -> Self: ...
    @overload
    def __getitem__(self, item: tuple[slice, Sequence[int]]) -> Self: ...
    @overload
    def __getitem__(self, item: tuple[Sequence[int], str]) -> Series: ...  # type: ignore[overload-overlap]
    @overload
    def __getitem__(self, item: tuple[slice, str]) -> Series: ...  # type: ignore[overload-overlap]
    @overload
    def __getitem__(self, item: tuple[Sequence[int], Sequence[str]]) -> Self: ...
    @overload
    def __getitem__(self, item: tuple[slice, Sequence[str]]) -> Self: ...
    @overload
    def __getitem__(self, item: tuple[Sequence[int], int]) -> Series: ...  # type: ignore[overload-overlap]
    @overload
    def __getitem__(self, item: tuple[slice, int]) -> Series: ...  # type: ignore[overload-overlap]

    @overload
    def __getitem__(self, item: Sequence[int]) -> Self: ...

    @overload
    def __getitem__(self, item: str) -> Series: ...  # type: ignore[overload-overlap]

    @overload
    def __getitem__(self, item: Sequence[str]) -> Self: ...

    @overload
    def __getitem__(self, item: slice) -> Self: ...

    @overload
    def __getitem__(self, item: tuple[slice, slice]) -> Self: ...

    def __getitem__(self, item: Any) -> Any:
        return super().__getitem__(item)

    def lazy(self) -> LazyFrame[Any]:
        """
        Lazify the DataFrame (if possible).

        If a library does not support lazy execution, then this is a no-op.

        Examples:
            Construct pandas, Polars and PyArrow DataFrames:

            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals.stable.v1 as nw
            >>> df = {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)
            >>> df_pa = pa.table(df)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.lazy()

            Note that then, pandas and pyarrow dataframe stay eager, but Polars DataFrame becomes a Polars LazyFrame:

            >>> func(df_pd)
               foo  bar ham
            0    1  6.0   a
            1    2  7.0   b
            2    3  8.0   c
            >>> func(df_pl)
            <LazyFrame ...>
            >>> func(df_pa)
            pyarrow.Table
            foo: int64
            bar: double
            ham: string
            ----
            foo: [[1,2,3]]
            bar: [[6,7,8]]
            ham: [["a","b","c"]]
        """
        return super().lazy()  # type: ignore[return-value]

    # Not sure what mypy is complaining about, probably some fancy
    # thing that I need to understand category theory for
    @overload  # type: ignore[override]
    def to_dict(self, *, as_series: Literal[True] = ...) -> dict[str, Series]: ...
    @overload
    def to_dict(self, *, as_series: Literal[False]) -> dict[str, list[Any]]: ...
    @overload
    def to_dict(self, *, as_series: bool) -> dict[str, Series] | dict[str, list[Any]]: ...
    def to_dict(
        self, *, as_series: bool = True
    ) -> dict[str, Series] | dict[str, list[Any]]:
        """
        Convert DataFrame to a dictionary mapping column name to values.

        Arguments:
            as_series: If set to true ``True``, then the values are Narwhals Series,
                        otherwise the values are Any.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import pyarrow as pa
            >>> import narwhals.stable.v1 as nw
            >>> df = {
            ...     "A": [1, 2, 3, 4, 5],
            ...     "fruits": ["banana", "banana", "apple", "apple", "banana"],
            ...     "B": [5, 4, 3, 2, 1],
            ...     "animals": ["beetle", "fly", "beetle", "beetle", "beetle"],
            ...     "optional": [28, 300, None, 2, -30],
            ... }
            >>> df_pd = pd.DataFrame(df)
            >>> df_pl = pl.DataFrame(df)
            >>> df_pa = pa.table(df)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.to_dict(as_series=False)

            We can then pass either pandas, Polars or PyArrow to `func`:

            >>> func(df_pd)
            {'A': [1, 2, 3, 4, 5], 'fruits': ['banana', 'banana', 'apple', 'apple', 'banana'], 'B': [5, 4, 3, 2, 1], 'animals': ['beetle', 'fly', 'beetle', 'beetle', 'beetle'], 'optional': [28.0, 300.0, nan, 2.0, -30.0]}
            >>> func(df_pl)
            {'A': [1, 2, 3, 4, 5], 'fruits': ['banana', 'banana', 'apple', 'apple', 'banana'], 'B': [5, 4, 3, 2, 1], 'animals': ['beetle', 'fly', 'beetle', 'beetle', 'beetle'], 'optional': [28, 300, None, 2, -30]}
            >>> func(df_pa)
            {'A': [1, 2, 3, 4, 5], 'fruits': ['banana', 'banana', 'apple', 'apple', 'banana'], 'B': [5, 4, 3, 2, 1], 'animals': ['beetle', 'fly', 'beetle', 'beetle', 'beetle'], 'optional': [28, 300, None, 2, -30]}
        """
        return super().to_dict(as_series=as_series)  # type: ignore[return-value]

    def is_duplicated(self: Self) -> Series:
        r"""
        Get a mask of all duplicated rows in this DataFrame.

        Examples:
            >>> import narwhals.stable.v1 as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> df_pd = pd.DataFrame(
            ...     {
            ...         "a": [1, 2, 3, 1],
            ...         "b": ["x", "y", "z", "x"],
            ...     }
            ... )
            >>> df_pl = pl.DataFrame(
            ...     {
            ...         "a": [1, 2, 3, 1],
            ...         "b": ["x", "y", "z", "x"],
            ...     }
            ... )

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.is_duplicated()

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)  # doctest: +NORMALIZE_WHITESPACE
            0     True
            1    False
            2    False
            3     True
            dtype: bool

            >>> func(df_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [bool]
            [
                true
                false
                false
                true
            ]
        """
        return super().is_duplicated()  # type: ignore[return-value]

    def is_unique(self: Self) -> Series:
        r"""
        Get a mask of all unique rows in this DataFrame.

        Examples:
            >>> import narwhals.stable.v1 as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> df_pd = pd.DataFrame(
            ...     {
            ...         "a": [1, 2, 3, 1],
            ...         "b": ["x", "y", "z", "x"],
            ...     }
            ... )
            >>> df_pl = pl.DataFrame(
            ...     {
            ...         "a": [1, 2, 3, 1],
            ...         "b": ["x", "y", "z", "x"],
            ...     }
            ... )

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.is_unique()

            We can then pass either pandas or Polars to `func`:

            >>> func(df_pd)  # doctest: +NORMALIZE_WHITESPACE
            0    False
            1     True
            2     True
            3    False
            dtype: bool

            >>> func(df_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [bool]
            [
                false
                 true
                 true
                false
            ]
        """
        return super().is_unique()  # type: ignore[return-value]

    def _l1_norm(self: Self) -> Self:
        """Private, just used to test the stable API."""
        return self.select(all()._l1_norm())


class LazyFrame(NwLazyFrame[IntoFrameT]):
    """
    Narwhals DataFrame, backed by a native dataframe.

    The native dataframe might be pandas.DataFrame, polars.LazyFrame, ...

    This class is not meant to be instantiated directly - instead, use
    `narwhals.from_native`.
    """

    @property
    def _dataframe(self) -> type[DataFrame[Any]]:
        return DataFrame

    def collect(self) -> DataFrame[Any]:
        r"""
        Materialize this LazyFrame into a DataFrame.

        Returns:
            DataFrame

        Examples:
            >>> import narwhals.stable.v1 as nw
            >>> import polars as pl
            >>> lf_pl = pl.LazyFrame(
            ...     {
            ...         "a": ["a", "b", "a", "b", "b", "c"],
            ...         "b": [1, 2, 3, 4, 5, 6],
            ...         "c": [6, 5, 4, 3, 2, 1],
            ...     }
            ... )
            >>> lf = nw.from_native(lf_pl)
            >>> lf
            ┌───────────────────────────────────────┐
            | Narwhals LazyFrame                    |
            | Use `.to_native` to see native output |
            └───────────────────────────────────────┘
            >>> df = lf.group_by("a").agg(nw.all().sum()).collect()
            >>> df.to_native().sort("a")
            shape: (3, 3)
            ┌─────┬─────┬─────┐
            │ a   ┆ b   ┆ c   │
            │ --- ┆ --- ┆ --- │
            │ str ┆ i64 ┆ i64 │
            ╞═════╪═════╪═════╡
            │ a   ┆ 4   ┆ 10  │
            │ b   ┆ 11  ┆ 10  │
            │ c   ┆ 6   ┆ 1   │
            └─────┴─────┴─────┘
        """
        return super().collect()  # type: ignore[return-value]

    def _l1_norm(self: Self) -> Self:
        """Private, just used to test the stable API."""
        return self.select(all()._l1_norm())


class Series(NwSeries):
    """
    Narwhals Series, backed by a native series.

    The native series might be pandas.Series, polars.Series, ...

    This class is not meant to be instantiated directly - instead, use
    `narwhals.from_native`, making sure to pass `allow_series=True` or
    `series_only=True`.
    """

    # We need to override any method which don't return Self so that type
    # annotations are correct.

    @property
    def _dataframe(self) -> type[DataFrame[Any]]:
        return DataFrame

    def to_frame(self) -> DataFrame[Any]:
        """
        Convert to dataframe.

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals.stable.v1 as nw
            >>> s = [1, 2, 3]
            >>> s_pd = pd.Series(s, name="a")
            >>> s_pl = pl.Series("a", s)

            We define a library agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.to_frame()

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)
               a
            0  1
            1  2
            2  3
            >>> func(s_pl)
            shape: (3, 1)
            ┌─────┐
            │ a   │
            │ --- │
            │ i64 │
            ╞═════╡
            │ 1   │
            │ 2   │
            │ 3   │
            └─────┘
        """
        return super().to_frame()  # type: ignore[return-value]

    def value_counts(
        self: Self,
        *,
        sort: bool = False,
        parallel: bool = False,
        name: str | None = None,
        normalize: bool = False,
    ) -> DataFrame[Any]:
        r"""
        Count the occurrences of unique values.

        Arguments:
            sort: Sort the output by count in descending order. If set to False (default),
                the order of the output is random.
            parallel: Execute the computation in parallel. Used for Polars only.
            name: Give the resulting count column a specific name; if `normalize` is True
                defaults to "proportion", otherwise defaults to "count".
            normalize: If true gives relative frequencies of the unique values

        Examples:
            >>> import narwhals.stable.v1 as nw
            >>> import pandas as pd
            >>> import polars as pl
            >>> s_pd = pd.Series([1, 1, 2, 3, 2], name="s")
            >>> s_pl = pl.Series(values=[1, 1, 2, 3, 2], name="s")

            Let's define a dataframe-agnostic function:

            >>> @nw.narwhalify
            ... def func(s):
            ...     return s.value_counts(sort=True)

            We can then pass either pandas or Polars to `func`:

            >>> func(s_pd)  # doctest: +NORMALIZE_WHITESPACE
               s  count
            0  1      2
            1  2      2
            2  3      1

            >>> func(s_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3, 2)
            ┌─────┬───────┐
            │ s   ┆ count │
            │ --- ┆ ---   │
            │ i64 ┆ u32   │
            ╞═════╪═══════╡
            │ 1   ┆ 2     │
            │ 2   ┆ 2     │
            │ 3   ┆ 1     │
            └─────┴───────┘
        """
        return super().value_counts(  # type: ignore[return-value]
            sort=sort, parallel=parallel, name=name, normalize=normalize
        )


class Expr(NwExpr):
    def _l1_norm(self) -> Self:
        return super()._taxicab_norm()


class Schema(NwSchema):
    """
    Ordered mapping of column names to their data type.

    Arguments:
        schema: Mapping[str, DType] | Iterable[tuple[str, DType]] | None
            The schema definition given by column names and their associated.
            *instantiated* Narwhals data type. Accepts a mapping or an iterable of tuples.

    Examples:
        Define a schema by passing *instantiated* data types.

        >>> import narwhals.stable.v1 as nw
        >>> schema = nw.Schema({"foo": nw.Int8(), "bar": nw.String()})
        >>> schema  # doctest:+SKIP
        Schema({'foo': Int8, 'bar': String})

        Access the data type associated with a specific column name.

        >>> schema["foo"]
        Int8

        Access various schema properties using the `names`, `dtypes`, and `len` methods.

        >>> schema.names()
        ['foo', 'bar']
        >>> schema.dtypes()
        [Int8, String]
        >>> schema.len()
        2
    """


@overload
def _stableify(obj: NwDataFrame[IntoFrameT]) -> DataFrame[IntoFrameT]: ...
@overload
def _stableify(obj: NwLazyFrame[IntoFrameT]) -> LazyFrame[IntoFrameT]: ...
@overload
def _stableify(obj: NwSeries) -> Series: ...
@overload
def _stableify(obj: NwExpr) -> Expr: ...
@overload
def _stableify(obj: Any) -> Any: ...


def _stableify(
    obj: NwDataFrame[IntoFrameT] | NwLazyFrame[IntoFrameT] | NwSeries | NwExpr | Any,
) -> DataFrame[IntoFrameT] | LazyFrame[IntoFrameT] | Series | Expr | Any:
    if isinstance(obj, NwDataFrame):
        return DataFrame(
            obj._compliant_frame,
            level=obj._level,
        )
    if isinstance(obj, NwLazyFrame):
        return LazyFrame(
            obj._compliant_frame,
            level=obj._level,
        )
    if isinstance(obj, NwSeries):
        return Series(
            obj._compliant_series,
            level=obj._level,
        )
    if isinstance(obj, NwExpr):
        return Expr(obj._call)
    return obj


@overload
def from_native(
    native_dataframe: Any,
    *,
    strict: Literal[False],
    eager_only: None = ...,
    eager_or_interchange_only: Literal[True],
    series_only: None = ...,
    allow_series: Literal[True],
) -> Any: ...


@overload
def from_native(
    native_dataframe: Any,
    *,
    strict: Literal[False],
    eager_only: Literal[True],
    eager_or_interchange_only: None = ...,
    series_only: None = ...,
    allow_series: Literal[True],
) -> Any: ...


@overload
def from_native(
    native_dataframe: IntoDataFrameT,
    *,
    strict: Literal[False],
    eager_only: None = ...,
    eager_or_interchange_only: Literal[True],
    series_only: None = ...,
    allow_series: None = ...,
) -> DataFrame[IntoDataFrameT]: ...


@overload
def from_native(
    native_dataframe: T,
    *,
    strict: Literal[False],
    eager_only: None = ...,
    eager_or_interchange_only: Literal[True],
    series_only: None = ...,
    allow_series: None = ...,
) -> T: ...


@overload
def from_native(
    native_dataframe: IntoDataFrameT,
    *,
    strict: Literal[False],
    eager_only: Literal[True],
    eager_or_interchange_only: None = ...,
    series_only: None = ...,
    allow_series: None = ...,
) -> DataFrame[IntoDataFrameT]: ...


@overload
def from_native(
    native_dataframe: T,
    *,
    strict: Literal[False],
    eager_only: Literal[True],
    eager_or_interchange_only: None = ...,
    series_only: None = ...,
    allow_series: None = ...,
) -> T: ...


@overload
def from_native(
    native_dataframe: Any,
    *,
    strict: Literal[False],
    eager_only: None = ...,
    eager_or_interchange_only: None = ...,
    series_only: None = ...,
    allow_series: Literal[True],
) -> Any: ...


@overload
def from_native(
    native_dataframe: Any,
    *,
    strict: Literal[False],
    eager_only: None = ...,
    eager_or_interchange_only: None = ...,
    series_only: Literal[True],
    allow_series: None = ...,
) -> Any: ...


@overload
def from_native(
    native_dataframe: IntoFrameT,
    *,
    strict: Literal[False],
    eager_only: None = ...,
    eager_or_interchange_only: None = ...,
    series_only: None = ...,
    allow_series: None = ...,
) -> DataFrame[IntoFrameT] | LazyFrame[IntoFrameT]: ...


@overload
def from_native(
    native_dataframe: T,
    *,
    strict: Literal[False],
    eager_only: None = ...,
    eager_or_interchange_only: None = ...,
    series_only: None = ...,
    allow_series: None = ...,
) -> T: ...


@overload
def from_native(
    native_dataframe: IntoDataFrameT,
    *,
    strict: Literal[True] = ...,
    eager_only: None = ...,
    eager_or_interchange_only: Literal[True],
    series_only: None = ...,
    allow_series: None = ...,
) -> DataFrame[IntoDataFrameT]:
    """
    from_native(df, strict=True, eager_or_interchange_only=True)
    from_native(df, eager_or_interchange_only=True)
    """


@overload
def from_native(
    native_dataframe: IntoDataFrameT,
    *,
    strict: Literal[True] = ...,
    eager_only: Literal[True],
    eager_or_interchange_only: None = ...,
    series_only: None = ...,
    allow_series: None = ...,
) -> DataFrame[IntoDataFrameT]:
    """
    from_native(df, strict=True, eager_only=True)
    from_native(df, eager_only=True)
    """


@overload
def from_native(
    native_dataframe: Any,
    *,
    strict: Literal[True] = ...,
    eager_only: None = ...,
    eager_or_interchange_only: None = ...,
    series_only: None = ...,
    allow_series: Literal[True],
) -> DataFrame[Any] | LazyFrame[Any] | Series:
    """
    from_native(df, strict=True, allow_series=True)
    from_native(df, allow_series=True)
    """


@overload
def from_native(
    native_dataframe: Any,
    *,
    strict: Literal[True] = ...,
    eager_only: None = ...,
    eager_or_interchange_only: None = ...,
    series_only: Literal[True],
    allow_series: None = ...,
) -> Series:
    """
    from_native(df, strict=True, series_only=True)
    from_native(df, series_only=True)
    """


@overload
def from_native(
    native_dataframe: IntoFrameT,
    *,
    strict: Literal[True] = ...,
    eager_only: None = ...,
    eager_or_interchange_only: None = ...,
    series_only: None = ...,
    allow_series: None = ...,
) -> DataFrame[IntoFrameT] | LazyFrame[IntoFrameT]:
    """
    from_native(df, strict=True)
    from_native(df)
    """


# All params passed in as variables
@overload
def from_native(
    native_dataframe: Any,
    *,
    strict: bool,
    eager_only: bool | None,
    eager_or_interchange_only: bool | None = None,
    series_only: bool | None,
    allow_series: bool | None,
) -> Any: ...


def from_native(
    native_dataframe: Any,
    *,
    strict: bool = True,
    eager_only: bool | None = None,
    eager_or_interchange_only: bool | None = None,
    series_only: bool | None = None,
    allow_series: bool | None = None,
) -> Any:
    """
    Convert dataframe/series to Narwhals DataFrame, LazyFrame, or Series.

    Arguments:
        native_dataframe: Raw object from user.
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
        eager_or_interchange_only: Whether to only allow eager objects or objects which
            implement the Dataframe Interchange Protocol.
        series_only: Whether to only allow series.
        allow_series: Whether to allow series (default is only dataframe / lazyframe).

    Returns:
        narwhals.DataFrame or narwhals.LazyFrame or narwhals.Series
    """
    from narwhals.stable.v1 import dtypes

    # Early returns
    if isinstance(native_dataframe, (DataFrame, LazyFrame)) and not series_only:
        return native_dataframe
    if isinstance(native_dataframe, Series) and (series_only or allow_series):
        return native_dataframe
    result = _from_native_impl(
        native_dataframe,
        strict=strict,
        eager_only=eager_only,
        eager_or_interchange_only=eager_or_interchange_only,
        series_only=series_only,
        allow_series=allow_series,
        dtypes=dtypes,  # type: ignore[arg-type]
    )
    return _stableify(result)


def narwhalify(
    func: Callable[..., Any] | None = None,
    *,
    strict: bool = False,
    eager_only: bool | None = False,
    eager_or_interchange_only: bool | None = False,
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
    import narwhals.stable.v1 as nw


    def func(df):
        df = nw.from_native(df, strict=False)
        df = df.group_by("a").agg(nw.col("b").sum())
        return nw.to_native(df)
    ```

    you can just write

    ```python
    import narwhals.stable.v1 as nw


    @nw.narwhalify
    def func(df):
        return df.group_by("a").agg(nw.col("b").sum())
    ```

    You can also pass in extra arguments, e.g.

    ```python
    @nw.narwhalify(eager_only=True)
    ```

    that will get passed down to `nw.from_native`.

    Arguments:
        func: Function to wrap in a `from_native`-`to_native` block.
        strict: Whether to raise if object can't be converted or to just leave it as-is
            (default).
        eager_only: Whether to only allow eager objects.
        eager_or_interchange_only: Whether to only allow eager objects or objects which
            implement the Dataframe Interchange Protocol.
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
                    eager_or_interchange_only=eager_or_interchange_only,
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
                    eager_or_interchange_only=eager_or_interchange_only,
                    series_only=series_only,
                    allow_series=allow_series,
                )
                for name, value in kwargs.items()
            }

            backends = {
                b()
                for v in (*args, *kwargs.values())
                if (b := getattr(v, "__native_namespace__", None))
            }

            if backends.__len__() > 1:
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


def all() -> Expr:
    """
    Instantiate an expression representing all columns.

    Examples:
        >>> import polars as pl
        >>> import pandas as pd
        >>> import narwhals.stable.v1 as nw
        >>> df_pd = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        >>> df_pl = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        Let's define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.select(nw.all() * 2)

        We can then pass either pandas or Polars to `func`:

        >>> func(df_pd)
           a   b
        0  2   8
        1  4  10
        2  6  12
        >>> func(df_pl)
        shape: (3, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 2   ┆ 8   │
        │ 4   ┆ 10  │
        │ 6   ┆ 12  │
        └─────┴─────┘
    """
    return _stableify(nw.all())


def col(*names: str | Iterable[str]) -> Expr:
    """
    Creates an expression that references one or more columns by their name(s).

    Arguments:
        names: Name(s) of the columns to use in the aggregation function.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals.stable.v1 as nw
        >>> df_pl = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        >>> df_pd = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

        We define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.select(nw.col("a") * nw.col("b"))

        We can then pass either pandas or polars to `func`:

        >>> func(df_pd)
           a
        0  3
        1  8
        >>> func(df_pl)
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 3   │
        │ 8   │
        └─────┘
    """
    return _stableify(nw.col(*names))


def nth(*indices: int | Sequence[int]) -> Expr:
    """
    Creates an expression that references one or more columns by their index(es).

    Notes:
        `nth` is not supported for Polars version<1.0.0. Please use [`col`](/api-reference/narwhals/#narwhals.col) instead.

    Arguments:
        indices: One or more indices representing the columns to retrieve.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals.stable.v1 as nw
        >>> data = {"a": [1, 2], "b": [3, 4]}
        >>> df_pl = pl.DataFrame(data)
        >>> df_pd = pd.DataFrame(data)
        >>> df_pa = pa.table(data)

        We define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.select(nw.nth(0) * 2)

        We can then pass either pandas or polars to `func`:

        >>> func(df_pd)
           a
        0  2
        1  4
        >>> func(df_pl)  # doctest: +SKIP
        shape: (2, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 2   │
        │ 4   │
        └─────┘
        >>> func(df_pa)
        pyarrow.Table
        a: int64
        ----
        a: [[2,4]]
    """
    return _stableify(nw.nth(*indices))


def len() -> Expr:
    """
    Return the number of rows.

    Examples:
        >>> import polars as pl
        >>> import pandas as pd
        >>> import narwhals.stable.v1 as nw
        >>> df_pd = pd.DataFrame({"a": [1, 2], "b": [5, 10]})
        >>> df_pl = pl.DataFrame({"a": [1, 2], "b": [5, 10]})

        Let's define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.select(nw.len())

        We can then pass either pandas or Polars to `func`:

        >>> func(df_pd)
           len
        0    2
        >>> func(df_pl)
        shape: (1, 1)
        ┌─────┐
        │ len │
        │ --- │
        │ u32 │
        ╞═════╡
        │ 2   │
        └─────┘
    """
    return _stableify(nw.len())


def lit(value: Any, dtype: DType | None = None) -> Expr:
    """
    Return an expression representing a literal value.

    Arguments:
        value: The value to use as literal.
        dtype: The data type of the literal value. If not provided, the data type will be inferred.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals.stable.v1 as nw
        >>> df_pl = pl.DataFrame({"a": [1, 2]})
        >>> df_pd = pd.DataFrame({"a": [1, 2]})

        We define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.with_columns(nw.lit(3).alias("b"))

        We can then pass either pandas or polars to `func`:

        >>> func(df_pd)
           a  b
        0  1  3
        1  2  3
        >>> func(df_pl)
        shape: (2, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i32 │
        ╞═════╪═════╡
        │ 1   ┆ 3   │
        │ 2   ┆ 3   │
        └─────┴─────┘

    """
    return _stableify(nw.lit(value, dtype))


def min(*columns: str) -> Expr:
    """
    Return the minimum value.

    Note:
       Syntactic sugar for ``nw.col(columns).min()``.

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function.

    Examples:
        >>> import polars as pl
        >>> import pandas as pd
        >>> import narwhals.stable.v1 as nw
        >>> df_pd = pd.DataFrame({"a": [1, 2], "b": [5, 10]})
        >>> df_pl = pl.DataFrame({"a": [1, 2], "b": [5, 10]})

        Let's define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.select(nw.min("b"))

        We can then pass either pandas or Polars to `func`:

        >>> func(df_pd)
           b
        0  5
        >>> func(df_pl)
        shape: (1, 1)
        ┌─────┐
        │ b   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 5   │
        └─────┘
    """
    return _stableify(nw.min(*columns))


def max(*columns: str) -> Expr:
    """
    Return the maximum value.

    Note:
       Syntactic sugar for ``nw.col(columns).max()``.

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function.

    Examples:
        >>> import polars as pl
        >>> import pandas as pd
        >>> import narwhals.stable.v1 as nw
        >>> df_pd = pd.DataFrame({"a": [1, 2], "b": [5, 10]})
        >>> df_pl = pl.DataFrame({"a": [1, 2], "b": [5, 10]})

        Let's define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.select(nw.max("a"))

        We can then pass either pandas or Polars to `func`:

        >>> func(df_pd)
           a
        0  2
        >>> func(df_pl)
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 2   │
        └─────┘
    """
    return _stableify(nw.max(*columns))


def mean(*columns: str) -> Expr:
    """
    Get the mean value.

    Note:
        Syntactic sugar for ``nw.col(columns).mean()``

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals.stable.v1 as nw
        >>> df_pl = pl.DataFrame({"a": [1, 8, 3]})
        >>> df_pd = pd.DataFrame({"a": [1, 8, 3]})

        We define a dataframe agnostic function:

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.select(nw.mean("a"))

        We can then pass either pandas or Polars to `func`:

        >>> func(df_pd)
             a
        0  4.0
        >>> func(df_pl)
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 4.0 │
        └─────┘
    """
    return _stableify(nw.mean(*columns))


def sum(*columns: str) -> Expr:
    """
    Sum all values.

    Note:
        Syntactic sugar for ``nw.col(columns).sum()``

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals.stable.v1 as nw
        >>> df_pl = pl.DataFrame({"a": [1, 2]})
        >>> df_pd = pd.DataFrame({"a": [1, 2]})

        We define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.select(nw.sum("a"))

        We can then pass either pandas or polars to `func`:

        >>> func(df_pd)
           a
        0  3
        >>> func(df_pl)
        shape: (1, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 3   │
        └─────┘
    """
    return _stableify(nw.sum(*columns))


def sum_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """
    Sum all values horizontally across columns.

    Warning:
        Unlike Polars, we support horizontal sum over numeric columns only.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals.stable.v1 as nw
        >>> data = {"a": [1, 2, 3], "b": [5, 10, None]}
        >>> df_pl = pl.DataFrame(data)
        >>> df_pd = pd.DataFrame(data)

        We define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.select(nw.sum_horizontal("a", "b"))

        We can then pass either pandas or polars to `func`:

        >>> func(df_pd)
              a
        0   6.0
        1  12.0
        2   3.0
        >>> func(df_pl)
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ i64 │
        ╞═════╡
        │ 6   │
        │ 12  │
        │ 3   │
        └─────┘
    """
    return _stableify(nw.sum_horizontal(*exprs))


def all_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    r"""
    Compute the bitwise AND horizontally across columns.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts expression input.

    Notes:
        pandas and Polars handle null values differently.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals.stable.v1 as nw
        >>> data = {
        ...     "a": [False, False, True, True, False, None],
        ...     "b": [False, True, True, None, None, None],
        ... }
        >>> df_pl = pl.DataFrame(data)
        >>> df_pd = pd.DataFrame(data)

        We define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.select("a", "b", all=nw.all_horizontal("a", "b"))

        We can then pass either pandas or polars to `func`:

        >>> func(df_pd)
               a      b    all
        0  False  False  False
        1  False   True  False
        2   True   True   True
        3   True   None  False
        4  False   None  False
        5   None   None  False

        >>> func(df_pl)
        shape: (6, 3)
        ┌───────┬───────┬───────┐
        │ a     ┆ b     ┆ all   │
        │ ---   ┆ ---   ┆ ---   │
        │ bool  ┆ bool  ┆ bool  │
        ╞═══════╪═══════╪═══════╡
        │ false ┆ false ┆ false │
        │ false ┆ true  ┆ false │
        │ true  ┆ true  ┆ true  │
        │ true  ┆ null  ┆ null  │
        │ false ┆ null  ┆ false │
        │ null  ┆ null  ┆ null  │
        └───────┴───────┴───────┘
    """
    return _stableify(nw.all_horizontal(*exprs))


def any_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    r"""
    Compute the bitwise OR horizontally across columns.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts expression input.

    Notes:
        pandas and Polars handle null values differently.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals.stable.v1 as nw
        >>> data = {
        ...     "a": [False, False, True, True, False, None],
        ...     "b": [False, True, True, None, None, None],
        ... }
        >>> df_pl = pl.DataFrame(data)
        >>> df_pd = pd.DataFrame(data)

        We define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.select("a", "b", any=nw.any_horizontal("a", "b"))

        We can then pass either pandas or polars to `func`:

        >>> func(df_pd)
               a      b    any
        0  False  False  False
        1  False   True   True
        2   True   True   True
        3   True   None   True
        4  False   None  False
        5   None   None  False

        >>> func(df_pl)
        shape: (6, 3)
        ┌───────┬───────┬───────┐
        │ a     ┆ b     ┆ any   │
        │ ---   ┆ ---   ┆ ---   │
        │ bool  ┆ bool  ┆ bool  │
        ╞═══════╪═══════╪═══════╡
        │ false ┆ false ┆ false │
        │ false ┆ true  ┆ true  │
        │ true  ┆ true  ┆ true  │
        │ true  ┆ null  ┆ true  │
        │ false ┆ null  ┆ null  │
        │ null  ┆ null  ┆ null  │
        └───────┴───────┴───────┘
    """
    return _stableify(nw.any_horizontal(*exprs))


def mean_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """
    Compute the mean of all values horizontally across columns.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals.stable.v1 as nw
        >>> data = {
        ...     "a": [1, 8, 3],
        ...     "b": [4, 5, None],
        ...     "c": ["x", "y", "z"],
        ... }
        >>> df_pl = pl.DataFrame(data)
        >>> df_pd = pd.DataFrame(data)

        We define a dataframe-agnostic function that computes the horizontal mean of "a"
        and "b" columns:

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.select(nw.mean_horizontal("a", "b"))

        We can then pass either pandas or polars to `func`:

        >>> func(df_pd)
             a
        0  2.5
        1  6.5
        2  3.0
        >>> func(df_pl)
        shape: (3, 1)
        ┌─────┐
        │ a   │
        │ --- │
        │ f64 │
        ╞═════╡
        │ 2.5 │
        │ 6.5 │
        │ 3.0 │
        └─────┘
    """
    return _stableify(nw.mean_horizontal(*exprs))


@overload
def concat(
    items: Iterable[DataFrame[Any]],
    *,
    how: Literal["horizontal", "vertical"] = "vertical",
) -> DataFrame[Any]: ...


@overload
def concat(
    items: Iterable[LazyFrame[Any]],
    *,
    how: Literal["horizontal", "vertical"] = "vertical",
) -> LazyFrame[Any]: ...


def concat(
    items: Iterable[DataFrame[Any] | LazyFrame[Any]],
    *,
    how: Literal["horizontal", "vertical"] = "vertical",
) -> DataFrame[Any] | LazyFrame[Any]:
    """
    Concatenate multiple DataFrames, LazyFrames into a single entity.

    Arguments:
        items: DataFrames, LazyFrames to concatenate.

        how: {'vertical', 'horizontal'}
            * vertical: Stacks Series from DataFrames vertically and fills with `null`
              if the lengths don't match.
            * horizontal: Stacks Series from DataFrames horizontally and fills with `null`
              if the lengths don't match.

    Returns:
        A new DataFrame, Lazyframe resulting from the concatenation.

    Raises:
        NotImplementedError: The items to concatenate should either all be eager, or all lazy

    Examples:

        Let's take an example of vertical concatenation:

        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals.stable.v1 as nw
        >>> data_1 = {"a": [1, 2, 3], "b": [4, 5, 6]}
        >>> data_2 = {"a": [5, 2], "b": [1, 4]}

        >>> df_pd_1 = pd.DataFrame(data_1)
        >>> df_pd_2 = pd.DataFrame(data_2)
        >>> df_pl_1 = pl.DataFrame(data_1)
        >>> df_pl_2 = pl.DataFrame(data_2)

        Let's define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df1, df2):
        ...     return nw.concat([df1, df2], how="vertical")

        >>> func(df_pd_1, df_pd_2)
           a  b
        0  1  4
        1  2  5
        2  3  6
        0  5  1
        1  2  4
        >>> func(df_pl_1, df_pl_2)
        shape: (5, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 5   │
        │ 3   ┆ 6   │
        │ 5   ┆ 1   │
        │ 2   ┆ 4   │
        └─────┴─────┘

        Let's look at case a for horizontal concatenation:

        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals.stable.v1 as nw
        >>> data_1 = {"a": [1, 2, 3], "b": [4, 5, 6]}
        >>> data_2 = {"c": [5, 2], "d": [1, 4]}

        >>> df_pd_1 = pd.DataFrame(data_1)
        >>> df_pd_2 = pd.DataFrame(data_2)
        >>> df_pl_1 = pl.DataFrame(data_1)
        >>> df_pl_2 = pl.DataFrame(data_2)

        Defining a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df1, df2):
        ...     return nw.concat([df1, df2], how="horizontal")

        >>> func(df_pd_1, df_pd_2)
           a  b    c    d
        0  1  4  5.0  1.0
        1  2  5  2.0  4.0
        2  3  6  NaN  NaN

        >>> func(df_pl_1, df_pl_2)
        shape: (3, 4)
        ┌─────┬─────┬──────┬──────┐
        │ a   ┆ b   ┆ c    ┆ d    │
        │ --- ┆ --- ┆ ---  ┆ ---  │
        │ i64 ┆ i64 ┆ i64  ┆ i64  │
        ╞═════╪═════╪══════╪══════╡
        │ 1   ┆ 4   ┆ 5    ┆ 1    │
        │ 2   ┆ 5   ┆ 2    ┆ 4    │
        │ 3   ┆ 6   ┆ null ┆ null │
        └─────┴─────┴──────┴──────┘

    """
    return _stableify(nw.concat(items, how=how))  # type: ignore[no-any-return]


def concat_str(
    exprs: IntoExpr | Iterable[IntoExpr],
    *more_exprs: IntoExpr,
    separator: str = "",
    ignore_nulls: bool = False,
) -> Expr:
    r"""
    Horizontally concatenate columns into a single string column.

    Arguments:
        exprs: Columns to concatenate into a single string column. Accepts expression
            input. Strings are parsed as column names, other non-expression inputs are
            parsed as literals. Non-`String` columns are cast to `String`.
        *more_exprs: Additional columns to concatenate into a single string column,
            specified as positional arguments.
        separator: String that will be used to separate the values of each column.
        ignore_nulls: Ignore null values (default is `False`).
            If set to `False`, null values will be propagated and if the row contains any
            null values, the output is null.

    Examples:
        >>> import narwhals.stable.v1 as nw
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> data = {
        ...     "a": [1, 2, 3],
        ...     "b": ["dogs", "cats", None],
        ...     "c": ["play", "swim", "walk"],
        ... }

        We define a dataframe-agnostic function that computes the horizontal string
        concatenation of different columns

        >>> @nw.narwhalify
        ... def func(df):
        ...     return df.select(
        ...         nw.concat_str(
        ...             [
        ...                 nw.col("a") * 2,
        ...                 nw.col("b"),
        ...                 nw.col("c"),
        ...             ],
        ...             separator=" ",
        ...         ).alias("full_sentence")
        ...     )

        We can then pass either pandas, Polars or PyArrow to `func`:

        >>> func(pd.DataFrame(data))
          full_sentence
        0   2 dogs play
        1   4 cats swim
        2          None

        >>> func(pl.DataFrame(data))
        shape: (3, 1)
        ┌───────────────┐
        │ full_sentence │
        │ ---           │
        │ str           │
        ╞═══════════════╡
        │ 2 dogs play   │
        │ 4 cats swim   │
        │ null          │
        └───────────────┘

        >>> func(pa.table(data))
        pyarrow.Table
        full_sentence: string
        ----
        full_sentence: [["2 dogs play","4 cats swim",null]]
    """

    return _stableify(
        nw.concat_str(exprs, *more_exprs, separator=separator, ignore_nulls=ignore_nulls)
    )


def is_ordered_categorical(series: Series) -> bool:
    """
    Return whether indices of categories are semantically meaningful.

    This is a convenience function to accessing what would otherwise be
    the `is_ordered` property from the DataFrame Interchange Protocol,
    see https://data-apis.org/dataframe-protocol/latest/API.html.

    - For Polars:
      - Enums are always ordered.
      - Categoricals are ordered if `dtype.ordering == "physical"`.
    - For pandas-like APIs:
      - Categoricals are ordered if `dtype.cat.ordered == True`.
    - For PyArrow table:
      - Categoricals are ordered if `dtype.type.ordered == True`.

    Examples:
        >>> import narwhals.stable.v1 as nw
        >>> import pandas as pd
        >>> import polars as pl
        >>> data = ["x", "y"]
        >>> s_pd = pd.Series(data, dtype=pd.CategoricalDtype(ordered=True))
        >>> s_pl = pl.Series(data, dtype=pl.Categorical(ordering="physical"))

        Let's define a library-agnostic function:

        >>> @nw.narwhalify
        ... def func(s):
        ...     return nw.is_ordered_categorical(s)

        Then, we can pass any supported library to `func`:

        >>> func(s_pd)
        True
        >>> func(s_pl)
        True
    """
    return nw_is_ordered_categorical(series)


def maybe_align_index(lhs: T, rhs: Series | DataFrame[Any] | LazyFrame[Any]) -> T:
    """
    Align `lhs` to the Index of `rhs`, if they're both pandas-like.

    Notes:
        This is only really intended for backwards-compatibility purposes,
        for example if your library already aligns indices for users.
        If you're designing a new library, we highly encourage you to not
        rely on the Index.
        For non-pandas-like inputs, this only checks that `lhs` and `rhs`
        are the same length.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals.stable.v1 as nw
        >>> df_pd = pd.DataFrame({"a": [1, 2]}, index=[3, 4])
        >>> s_pd = pd.Series([6, 7], index=[4, 3])
        >>> df = nw.from_native(df_pd)
        >>> s = nw.from_native(s_pd, series_only=True)
        >>> nw.to_native(nw.maybe_align_index(df, s))
           a
        4  2
        3  1
    """
    return nw_maybe_align_index(lhs, rhs)


def maybe_convert_dtypes(df: T, *args: bool, **kwargs: bool | str) -> T:
    """
    Convert columns or series to the best possible dtypes using dtypes supporting ``pd.NA``, if df is pandas-like.

    Arguments:
        obj: DataFrame or Series.
        *args: Additional arguments which gets passed through.
        **kwargs: Additional arguments which gets passed through.

    Notes:
        For non-pandas-like inputs, this is a no-op.
        Also, `args` and `kwargs` just get passed down to the underlying library as-is.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals.stable.v1 as nw
        >>> import numpy as np
        >>> df_pd = pd.DataFrame(
        ...     {
        ...         "a": pd.Series([1, 2, 3], dtype=np.dtype("int32")),
        ...         "b": pd.Series([True, False, np.nan], dtype=np.dtype("O")),
        ...     }
        ... )
        >>> df = nw.from_native(df_pd)
        >>> nw.to_native(nw.maybe_convert_dtypes(df)).dtypes  # doctest: +NORMALIZE_WHITESPACE
        a             Int32
        b           boolean
        dtype: object
    """
    return nw_maybe_convert_dtypes(df, *args, **kwargs)


def maybe_get_index(obj: T) -> Any | None:
    """
    Get the index of a DataFrame or a Series, if it's pandas-like.

    Notes:
        This is only really intended for backwards-compatibility purposes,
        for example if your library already aligns indices for users.
        If you're designing a new library, we highly encourage you to not
        rely on the Index.
        For non-pandas-like inputs, this returns `None`.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals.stable.v1 as nw
        >>> df_pd = pd.DataFrame({"a": [1, 2], "b": [4, 5]})
        >>> df = nw.from_native(df_pd)
        >>> nw.maybe_get_index(df)
        RangeIndex(start=0, stop=2, step=1)
        >>> series_pd = pd.Series([1, 2])
        >>> series = nw.from_native(series_pd, series_only=True)
        >>> nw.maybe_get_index(series)
        RangeIndex(start=0, stop=2, step=1)
    """
    return nw_maybe_get_index(obj)


def maybe_set_index(df: T, column_names: str | list[str]) -> T:
    """
    Set columns `columns` to be the index of `df`, if `df` is pandas-like.

    Notes:
        This is only really intended for backwards-compatibility purposes,
        for example if your library already aligns indices for users.
        If you're designing a new library, we highly encourage you to not
        rely on the Index.
        For non-pandas-like inputs, this is a no-op.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals.stable.v1 as nw
        >>> df_pd = pd.DataFrame({"a": [1, 2], "b": [4, 5]})
        >>> df = nw.from_native(df_pd)
        >>> nw.to_native(nw.maybe_set_index(df, "b"))  # doctest: +NORMALIZE_WHITESPACE
           a
        b
        4  1
        5  2
    """
    return nw_maybe_set_index(df, column_names)


def maybe_reset_index(obj: T) -> T:
    """
    Reset the index to the default integer index of a DataFrame or a Series, if it's pandas-like.

    Notes:
        This is only really intended for backwards-compatibility purposes,
        for example if your library already resets the index for users.
        If you're designing a new library, we highly encourage you to not
        rely on the Index.
        For non-pandas-like inputs, this is a no-op.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals.stable.v1 as nw
        >>> df_pd = pd.DataFrame({"a": [1, 2], "b": [4, 5]}, index=([6, 7]))
        >>> df = nw.from_native(df_pd)
        >>> nw.to_native(nw.maybe_reset_index(df))
           a  b
        0  1  4
        1  2  5
        >>> series_pd = pd.Series([1, 2])
        >>> series = nw.from_native(series_pd, series_only=True)
        >>> nw.maybe_get_index(series)
        RangeIndex(start=0, stop=2, step=1)
    """
    return nw_maybe_reset_index(obj)


def get_native_namespace(obj: Any) -> Any:
    """
    Get native namespace from object.

    Examples:
        >>> import polars as pl
        >>> import pandas as pd
        >>> import narwhals.stable.v1 as nw
        >>> df = nw.from_native(pd.DataFrame({"a": [1, 2, 3]}))
        >>> nw.get_native_namespace(df)
        <module 'pandas'...>
        >>> df = nw.from_native(pl.DataFrame({"a": [1, 2, 3]}))
        >>> nw.get_native_namespace(df)
        <module 'polars'...>
    """
    return nw_get_native_namespace(obj)


def get_level(
    obj: DataFrame[Any] | LazyFrame[Any] | Series,
) -> Literal["full", "interchange"]:
    """
    Level of support Narwhals has for current object.

    This can be one of:

    - 'full': full Narwhals API support
    - 'metadata': only metadata operations are supported (`df.schema`)
    """
    return nw.get_level(obj)


class When(NwWhen):
    @classmethod
    def from_when(cls, when: NwWhen) -> Self:
        return cls(*when._predicates)

    def then(self, value: Any) -> Then:
        return Then.from_then(super().then(value))


class Then(NwThen, Expr):
    @classmethod
    def from_then(cls, then: NwThen) -> Self:
        return cls(then._call)

    def otherwise(self, value: Any) -> Expr:
        return _stableify(super().otherwise(value))


def when(*predicates: IntoExpr | Iterable[IntoExpr]) -> When:
    """
    Start a `when-then-otherwise` expression.

    Expression similar to an `if-else` statement in Python. Always initiated by a `pl.when(<condition>).then(<value if condition>)`., and optionally followed by chaining one or more `.when(<condition>).then(<value>)` statements.
    Chained when-then operations should be read as Python `if, elif, ... elif` blocks, not as `if, if, ... if`, i.e. the first condition that evaluates to `True` will be picked.
    If none of the conditions are `True`, an optional `.otherwise(<value if all statements are false>)` can be appended at the end. If not appended, and none of the conditions are `True`, `None` will be returned.

    Arguments:
        predicates: Condition(s) that must be met in order to apply the subsequent statement. Accepts one or more boolean expressions, which are implicitly combined with `&`. String input is parsed as a column name.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals.stable.v1 as nw
        >>> df_pl = pl.DataFrame({"a": [1, 2, 3], "b": [5, 10, 15]})
        >>> df_pd = pd.DataFrame({"a": [1, 2, 3], "b": [5, 10, 15]})

        We define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df_any):
        ...     return df_any.with_columns(
        ...         nw.when(nw.col("a") < 3).then(5).otherwise(6).alias("a_when")
        ...     )

        We can then pass either pandas or polars to `func`:

        >>> func(df_pd)
           a   b  a_when
        0  1   5       5
        1  2  10       5
        2  3  15       6
        >>> func(df_pl)
        shape: (3, 3)
        ┌─────┬─────┬────────┐
        │ a   ┆ b   ┆ a_when │
        │ --- ┆ --- ┆ ---    │
        │ i64 ┆ i64 ┆ i32    │
        ╞═════╪═════╪════════╡
        │ 1   ┆ 5   ┆ 5      │
        │ 2   ┆ 10  ┆ 5      │
        │ 3   ┆ 15  ┆ 6      │
        └─────┴─────┴────────┘
    """
    return When.from_when(nw_when(*predicates))


def new_series(
    name: str,
    values: Any,
    dtype: DType | type[DType] | None = None,
    *,
    native_namespace: ModuleType,
) -> Series:
    """
    Instantiate Narwhals Series from raw data.

    Arguments:
        name: Name of resulting Series.
        values: Values of make Series from.
        dtype: (Narwhals) dtype. If not provided, the native library
            may auto-infer it from `values`.
        native_namespace: The native library to use for DataFrame creation.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals.stable.v1 as nw
        >>> data = {"a": [1, 2, 3], "b": [4, 5, 6]}

        Let's define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df):
        ...     values = [4, 1, 2]
        ...     native_namespace = nw.get_native_namespace(df)
        ...     return nw.new_series("c", values, nw.Int32, native_namespace=native_namespace)

        Let's see what happens when passing pandas / Polars input:

        >>> func(pd.DataFrame(data))
        0    4
        1    1
        2    2
        Name: c, dtype: int32
        >>> func(pl.DataFrame(data))  # doctest: +NORMALIZE_WHITESPACE
        shape: (3,)
        Series: 'c' [i32]
        [
           4
           1
           2
        ]
    """
    from narwhals.stable.v1 import dtypes

    return _stableify(
        _new_series_impl(
            name,
            values,
            dtype,
            native_namespace=native_namespace,
            dtypes=dtypes,  # type: ignore[arg-type]
        )
    )


def from_dict(
    data: dict[str, Any],
    schema: dict[str, DType] | Schema | None = None,
    *,
    native_namespace: ModuleType | None = None,
) -> DataFrame[Any]:
    """
    Instantiate DataFrame from dictionary.

    Notes:
        For pandas-like dataframes, conversion to schema is applied after dataframe
        creation.

    Arguments:
        data: Dictionary to create DataFrame from.
        schema: The DataFrame schema as Schema or dict of {name: type}.
        native_namespace: The native library to use for DataFrame creation. Only
            necessary if inputs are not Narwhals Series.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals.stable.v1 as nw
        >>> data = {"a": [1, 2, 3], "b": [4, 5, 6]}

        Let's create a new dataframe of the same class as the dataframe we started with, from a dict of new data:

        >>> @nw.narwhalify
        ... def func(df):
        ...     new_data = {"c": [5, 2], "d": [1, 4]}
        ...     native_namespace = nw.get_native_namespace(df)
        ...     return nw.from_dict(new_data, native_namespace=native_namespace)

        Let's see what happens when passing pandas / Polars input:

        >>> func(pd.DataFrame(data))
           c  d
        0  5  1
        1  2  4
        >>> func(pl.DataFrame(data))
        shape: (2, 2)
        ┌─────┬─────┐
        │ c   ┆ d   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 5   ┆ 1   │
        │ 2   ┆ 4   │
        └─────┴─────┘
    """
    from narwhals.stable.v1 import dtypes

    return _stableify(
        _from_dict_impl(
            data,
            schema,
            native_namespace=native_namespace,
            dtypes=dtypes,  # type: ignore[arg-type]
        )
    )


__all__ = [
    "selectors",
    "concat",
    "dependencies",
    "to_native",
    "from_native",
    "is_ordered_categorical",
    "maybe_align_index",
    "maybe_convert_dtypes",
    "maybe_get_index",
    "maybe_reset_index",
    "maybe_set_index",
    "get_native_namespace",
    "get_level",
    "all",
    "all_horizontal",
    "any_horizontal",
    "col",
    "concat_str",
    "nth",
    "len",
    "lit",
    "min",
    "max",
    "mean",
    "mean_horizontal",
    "sum",
    "sum_horizontal",
    "when",
    "DataFrame",
    "LazyFrame",
    "Series",
    "Expr",
    "Int64",
    "Int32",
    "Int16",
    "Int8",
    "UInt64",
    "UInt32",
    "UInt16",
    "UInt8",
    "Float64",
    "Float32",
    "Boolean",
    "Object",
    "Unknown",
    "Categorical",
    "Enum",
    "String",
    "Datetime",
    "Duration",
    "Struct",
    "Array",
    "List",
    "Date",
    "narwhalify",
    "show_versions",
    "Schema",
    "from_dict",
    "new_series",
]
