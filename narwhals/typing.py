from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, TypeVar, Union

from narwhals._compliant import CompliantDataFrame, CompliantLazyFrame, CompliantSeries

if TYPE_CHECKING:
    import datetime as dt
    from collections.abc import Callable, Iterable, Sequence, Sized
    from decimal import Decimal
    from types import ModuleType

    import numpy as np
    import pyarrow as pa
    from pandas.api.extensions import ExtensionDtype as PandasDType
    from typing_extensions import TypeAlias

    from narwhals import dtypes
    from narwhals.dataframe import DataFrame, LazyFrame
    from narwhals.expr import Expr
    from narwhals.series import Series

    # All dataframes supported by Narwhals have a
    # `columns` property. Their similarities don't extend
    # _that_ much further unfortunately...
    class NativeFrame(Protocol):
        @property
        def columns(self) -> Any: ...

        def join(self, *args: Any, **kwargs: Any) -> Any: ...

    class NativeLazyFrame(NativeFrame, Protocol):
        def explain(self, *args: Any, **kwargs: Any) -> Any: ...

    class NativeSeries(Sized, Iterable[Any], Protocol):
        def filter(self, *args: Any, **kwargs: Any) -> Any: ...

    class DataFrameLike(Protocol):
        def __dataframe__(self, *args: Any, **kwargs: Any) -> Any: ...

    class SupportsNativeNamespace(Protocol):
        def __native_namespace__(self) -> ModuleType: ...

    # ruff: noqa: N802
    class DTypes(Protocol):
        @property
        def Decimal(self) -> type[dtypes.Decimal]: ...
        @property
        def Int128(self) -> type[dtypes.Int128]: ...
        @property
        def Int64(self) -> type[dtypes.Int64]: ...
        @property
        def Int32(self) -> type[dtypes.Int32]: ...
        @property
        def Int16(self) -> type[dtypes.Int16]: ...
        @property
        def Int8(self) -> type[dtypes.Int8]: ...
        @property
        def UInt128(self) -> type[dtypes.UInt128]: ...
        @property
        def UInt64(self) -> type[dtypes.UInt64]: ...
        @property
        def UInt32(self) -> type[dtypes.UInt32]: ...
        @property
        def UInt16(self) -> type[dtypes.UInt16]: ...
        @property
        def UInt8(self) -> type[dtypes.UInt8]: ...
        @property
        def Float64(self) -> type[dtypes.Float64]: ...
        @property
        def Float32(self) -> type[dtypes.Float32]: ...
        @property
        def String(self) -> type[dtypes.String]: ...
        @property
        def Boolean(self) -> type[dtypes.Boolean]: ...
        @property
        def Object(self) -> type[dtypes.Object]: ...
        @property
        def Categorical(self) -> type[dtypes.Categorical]: ...
        @property
        def Enum(self) -> type[dtypes.Enum]: ...
        @property
        def Datetime(self) -> type[dtypes.Datetime]: ...
        @property
        def Duration(self) -> type[dtypes.Duration]: ...
        @property
        def Date(self) -> type[dtypes.Date]: ...
        @property
        def Field(self) -> type[dtypes.Field]: ...
        @property
        def Struct(self) -> type[dtypes.Struct]: ...
        @property
        def List(self) -> type[dtypes.List]: ...
        @property
        def Array(self) -> type[dtypes.Array]: ...
        @property
        def Unknown(self) -> type[dtypes.Unknown]: ...
        @property
        def Time(self) -> type[dtypes.Time]: ...
        @property
        def Binary(self) -> type[dtypes.Binary]: ...


IntoExpr: TypeAlias = Union["Expr", str, "Series[Any]"]
"""Anything which can be converted to an expression.

Use this to mean "either a Narwhals expression, or something which can be converted
into one". For example, `exprs` in `DataFrame.select` is typed to accept `IntoExpr`,
as it can either accept a `nw.Expr` (e.g. `df.select(nw.col('a'))`) or a string
which will be interpreted as a `nw.Expr`, e.g. `df.select('a')`.
"""

IntoDataFrame: TypeAlias = Union["NativeFrame", "DataFrameLike"]
"""Anything which can be converted to a Narwhals DataFrame.

Use this if your function accepts a narwhalifiable object but doesn't care about its backend.

Examples:
    >>> import narwhals as nw
    >>> from narwhals.typing import IntoDataFrame
    >>> def agnostic_shape(df_native: IntoDataFrame) -> tuple[int, int]:
    ...     df = nw.from_native(df_native, eager_only=True)
    ...     return df.shape
"""

IntoLazyFrame: TypeAlias = "NativeLazyFrame"

IntoFrame: TypeAlias = Union["IntoDataFrame", "IntoLazyFrame"]
"""Anything which can be converted to a Narwhals DataFrame or LazyFrame.

Use this if your function can accept an object which can be converted to either
`nw.DataFrame` or `nw.LazyFrame` and it doesn't care about its backend.

Examples:
    >>> import narwhals as nw
    >>> from narwhals.typing import IntoFrame
    >>> def agnostic_columns(df_native: IntoFrame) -> list[str]:
    ...     df = nw.from_native(df_native)
    ...     return df.collect_schema().names()
"""

Frame: TypeAlias = Union["DataFrame[Any]", "LazyFrame[Any]"]
"""Narwhals DataFrame or Narwhals LazyFrame.

Use this if your function can work with either and your function doesn't care
about its backend.

Examples:
    >>> import narwhals as nw
    >>> from narwhals.typing import Frame
    >>> @nw.narwhalify
    ... def agnostic_columns(df: Frame) -> list[str]:
    ...     return df.columns
"""

IntoSeries: TypeAlias = "NativeSeries"
"""Anything which can be converted to a Narwhals Series.

Use this if your function can accept an object which can be converted to `nw.Series`
and it doesn't care about its backend.

Examples:
    >>> from typing import Any
    >>> import narwhals as nw
    >>> from narwhals.typing import IntoSeries
    >>> def agnostic_to_list(s_native: IntoSeries) -> list[Any]:
    ...     s = nw.from_native(s_native)
    ...     return s.to_list()
"""

IntoFrameT = TypeVar("IntoFrameT", bound="IntoFrame")
"""TypeVar bound to object convertible to Narwhals DataFrame or Narwhals LazyFrame.

Use this if your function accepts an object which is convertible to `nw.DataFrame`
or `nw.LazyFrame` and returns an object of the same type.

Examples:
    >>> import narwhals as nw
    >>> from narwhals.typing import IntoFrameT
    >>> def agnostic_func(df_native: IntoFrameT) -> IntoFrameT:
    ...     df = nw.from_native(df_native)
    ...     return df.with_columns(c=nw.col("a") + 1).to_native()
"""

IntoDataFrameT = TypeVar("IntoDataFrameT", bound="IntoDataFrame")
"""TypeVar bound to object convertible to Narwhals DataFrame.

Use this if your function accepts an object which can be converted to `nw.DataFrame`
and returns an object of the same class.

Examples:
    >>> import narwhals as nw
    >>> from narwhals.typing import IntoDataFrameT
    >>> def agnostic_func(df_native: IntoDataFrameT) -> IntoDataFrameT:
    ...     df = nw.from_native(df_native, eager_only=True)
    ...     return df.with_columns(c=df["a"] + 1).to_native()
"""

IntoLazyFrameT = TypeVar("IntoLazyFrameT", bound="IntoLazyFrame")

FrameT = TypeVar("FrameT", "DataFrame[Any]", "LazyFrame[Any]")
"""TypeVar bound to Narwhals DataFrame or Narwhals LazyFrame.

Use this if your function accepts either `nw.DataFrame` or `nw.LazyFrame` and returns
an object of the same kind.

Examples:
    >>> import narwhals as nw
    >>> from narwhals.typing import FrameT
    >>> @nw.narwhalify
    ... def agnostic_func(df: FrameT) -> FrameT:
    ...     return df.with_columns(c=nw.col("a") + 1)
"""

DataFrameT = TypeVar("DataFrameT", bound="DataFrame[Any]")
"""TypeVar bound to Narwhals DataFrame.

Use this if your function can accept a Narwhals DataFrame and returns a Narwhals
DataFrame backed by the same backend.

Examples:
    >>> import narwhals as nw
    >>> from narwhals.typing import DataFrameT
    >>> @nw.narwhalify
    >>> def func(df: DataFrameT) -> DataFrameT:
    ...     return df.with_columns(c=df["a"] + 1)
"""

LazyFrameT = TypeVar("LazyFrameT", bound="LazyFrame[Any]")
SeriesT = TypeVar("SeriesT", bound="Series[Any]")

IntoSeriesT = TypeVar("IntoSeriesT", bound="IntoSeries")
"""TypeVar bound to object convertible to Narwhals Series.

Use this if your function accepts an object which can be converted to `nw.Series`
and returns an object of the same class.

Examples:
    >>> import narwhals as nw
    >>> from narwhals.typing import IntoSeriesT
    >>> def agnostic_abs(s_native: IntoSeriesT) -> IntoSeriesT:
    ...     s = nw.from_native(s_native, series_only=True)
    ...     return s.abs().to_native()
"""

DTypeBackend: TypeAlias = 'Literal["pyarrow", "numpy_nullable"] | None'
SizeUnit: TypeAlias = Literal[
    "b",
    "kb",
    "mb",
    "gb",
    "tb",
    "bytes",
    "kilobytes",
    "megabytes",
    "gigabytes",
    "terabytes",
]

TimeUnit: TypeAlias = Literal["ns", "us", "ms", "s"]

AsofJoinStrategy: TypeAlias = Literal["backward", "forward", "nearest"]
"""Join strategy.

- *"backward"*: Selects the last row in the right DataFrame whose `on` key
    is less than or equal to the left's key.
- *"forward"*: Selects the first row in the right DataFrame whose `on` key
    is greater than or equal to the left's key.
- *"nearest"*: Search selects the last row in the right DataFrame whose value
    is nearest to the left's key.
"""

ClosedInterval: TypeAlias = Literal["left", "right", "none", "both"]
"""Define which sides of the interval are closed (inclusive)."""

ConcatMethod: TypeAlias = Literal["horizontal", "vertical", "diagonal"]
"""Concatenating strategy.

- *"vertical"*: Concatenate vertically. Column names must match.
- *"horizontal"*: Concatenate horizontally. If lengths don't match, then
    missing rows are filled with null values.
- *"diagonal"*: Finds a union between the column schemas and fills missing
    column values with null.
"""

FillNullStrategy: TypeAlias = Literal["forward", "backward"]
"""Strategy used to fill null values."""

JoinStrategy: TypeAlias = Literal["inner", "left", "full", "cross", "semi", "anti"]
"""Join strategy.

- *"inner"*: Returns rows that have matching values in both tables.
- *"left"*: Returns all rows from the left table, and the matched rows from
    the right table.
- *"full"*: Returns all rows in both dataframes, with the `suffix` appended to
    the right join keys.
- *"cross"*: Returns the Cartesian product of rows from both tables.
- *"semi"*: Filter rows that have a match in the right table.
- *"anti"*: Filter rows that do not have a match in the right table.
"""

PivotAgg: TypeAlias = Literal[
    "min", "max", "first", "last", "sum", "mean", "median", "len"
]
"""A predefined aggregate function string."""

RankMethod: TypeAlias = Literal["average", "min", "max", "dense", "ordinal"]
"""The method used to assign ranks to tied elements.

- *"average"*: The average of the ranks that would have been assigned to
    all the tied values is assigned to each value.
- *"min"*: The minimum of the ranks that would have been assigned to all
    the tied values is assigned to each value. (This is also referred to
    as "competition" ranking.)
- *"max"*: The maximum of the ranks that would have been assigned to all
    the tied values is assigned to each value.
- *"dense"*: Like "min", but the rank of the next highest element is
    assigned the rank immediately after those assigned to the tied elements.
- *"ordinal"*: All values are given a distinct rank, corresponding to the
    order that the values occur in the Series.
"""

RollingInterpolationMethod: TypeAlias = Literal[
    "nearest", "higher", "lower", "midpoint", "linear"
]
"""Interpolation method."""

UniqueKeepStrategy: TypeAlias = Literal["any", "first", "last", "none"]
"""Which of the duplicate rows to keep.

- *"any"*: Does not give any guarantee of which row is kept.
    This allows more optimizations.
- *"none"*: Don't keep duplicate rows.
- *"first"*: Keep first unique row.
- *"last"*: Keep last unique row.
"""

LazyUniqueKeepStrategy: TypeAlias = Literal["any", "none"]
"""Which of the duplicate rows to keep.

- *"any"*: Does not give any guarantee of which row is kept.
- *"none"*: Don't keep duplicate rows.
"""


_ShapeT = TypeVar("_ShapeT", bound="tuple[int, ...]")
_NDArray: TypeAlias = "np.ndarray[_ShapeT, Any]"
_1DArray: TypeAlias = "_NDArray[tuple[int]]"  # noqa: PYI042
_1DArrayInt: TypeAlias = "np.ndarray[tuple[int], np.dtype[np.integer[Any]]]"  # noqa: PYI042
_2DArray: TypeAlias = "_NDArray[tuple[int, int]]"  # noqa: PYI042, PYI047
_AnyDArray: TypeAlias = "_NDArray[tuple[int, ...]]"  # noqa: PYI047
_NumpyScalar: TypeAlias = "np.generic[Any]"
Into1DArray: TypeAlias = "_1DArray | _NumpyScalar"
"""A 1-dimensional `numpy.ndarray` or scalar that can be converted into one."""


NumericLiteral: TypeAlias = "int | float | Decimal"
TemporalLiteral: TypeAlias = "dt.date | dt.datetime | dt.time | dt.timedelta"
NonNestedLiteral: TypeAlias = (
    "NumericLiteral | TemporalLiteral | str | bool | bytes | None"
)
PythonLiteral: TypeAlias = "NonNestedLiteral | list[Any] | tuple[Any, ...]"

NonNestedDType: TypeAlias = "dtypes.NumericType | dtypes.TemporalType | dtypes.String | dtypes.Boolean | dtypes.Binary | dtypes.Categorical | dtypes.Unknown | dtypes.Object"
"""Any Narwhals DType that does not have required arguments."""

IntoDType: TypeAlias = "dtypes.DType | type[NonNestedDType]"
"""Anything that can be converted into a Narwhals DType.

Examples:
    >>> import polars as pl
    >>> import narwhals as nw
    >>> df_native = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    >>> df = nw.from_native(df_native)
    >>> df.select(
    ...     nw.col("a").cast(nw.Int32),
    ...     nw.col("b").cast(nw.String()).str.split(".").cast(nw.List(nw.Int8)),
    ... )
    ┌──────────────────┐
    |Narwhals DataFrame|
    |------------------|
    |shape: (3, 2)     |
    |┌─────┬──────────┐|
    |│ a   ┆ b        │|
    |│ --- ┆ ---      │|
    |│ i32 ┆ list[i8] │|
    |╞═════╪══════════╡|
    |│ 1   ┆ [4, 0]   │|
    |│ 2   ┆ [5, 0]   │|
    |│ 3   ┆ [6, 0]   │|
    |└─────┴──────────┘|
    └──────────────────┘
"""


# Annotations for `__getitem__` methods
_T = TypeVar("_T")
_Slice: TypeAlias = "slice[_T, Any, Any] | slice[Any, _T, Any] | slice[None, None, _T]"
_SliceNone: TypeAlias = "slice[None, None, None]"
# Index/column positions
SingleIndexSelector: TypeAlias = int
_SliceIndex: TypeAlias = "_Slice[int] | _SliceNone"
"""E.g. `[1:]` or `[:3]` or `[::2]`."""
SizedMultiIndexSelector: TypeAlias = "Sequence[int] | _T | _1DArrayInt"
MultiIndexSelector: TypeAlias = "_SliceIndex | SizedMultiIndexSelector[_T]"
# Labels/column names
SingleNameSelector: TypeAlias = str
_SliceName: TypeAlias = "_Slice[str] | _SliceNone"
SizedMultiNameSelector: TypeAlias = "Sequence[str] | _T | _1DArray"
MultiNameSelector: TypeAlias = "_SliceName | SizedMultiNameSelector[_T]"
# Mixed selectors
SingleColSelector: TypeAlias = "SingleIndexSelector | SingleNameSelector"
MultiColSelector: TypeAlias = "MultiIndexSelector[_T] | MultiNameSelector[_T]"


class ToPandasArrowKwds(TypedDict, total=False):
    """Keyword arguments to be passed to [`pyarrow.Table.to_pandas`].

    [`pyarrow.Table.to_pandas`]: https://arrow.apache.org/docs/python/generated/pyarrow.Table.html#pyarrow.Table.to_pandas
    """

    memory_pool: pa.MemoryPool | None
    categories: list[Any] | None
    """List of fields that should be returned as pandas.Categorical.

    Only applies to table-like data structures"""
    strings_to_categorical: bool
    zero_copy_only: bool
    """Raise an ArrowException if this function call would require copying the underlying data."""
    integer_object_nulls: bool
    """Cast integers with nulls to objects"""
    date_as_object: bool
    """Cast dates to objects.

    If False, convert to datetime64 dtype with the equivalent time unit (if supported).

    Note:
        in pandas version < 2.0, only datetime64[ns] conversion is supported.
    """
    timestamp_as_object: bool
    """Cast non-nanosecond timestamps (np.datetime64) to objects.

    This is useful in pandas version 1.x if you have timestamps that don't fit in the normal date range of nanosecond timestamps (1678 CE-2262 CE).
    Non-nanosecond timestamps are supported in pandas version 2.0.
    If False, all timestamps are converted to datetime64 dtype.
    """
    use_threads: bool
    deduplicate_objects: bool
    ignore_metadata: bool
    safe: bool
    """For certain data types, a cast is needed in order to store the data in a pandas DataFrame or Series (e.g. timestamps are always stored as nanoseconds in pandas).

    This option controls whether it is a safe cast or not."""
    split_blocks: bool
    """If True, generate one internal “block” for each column when creating a pandas.DataFrame from a RecordBatch or Table.

    While this can temporarily reduce memory note that various pandas operations can trigger “consolidation” which may balloon memory use.
    """
    self_destruct: bool
    """EXPERIMENTAL: If True, attempt to deallocate the originating Arrow memory while converting the Arrow object to pandas.

    If you use the object after calling to_pandas with this option it will crash your program.
    Note that you may not see always memory usage improvements.
    For example, if multiple columns share an underlying allocation, memory can't be freed until all columns are converted.
    """
    maps_as_pydicts: Literal["None", "lossy", "strict"] | None
    types_mapper: Callable[[pa.DataType], PandasDType | None] | None
    """Used to override the default pandas type for conversion of built-in pyarrow types or in absence of pandas_metadata in the Table schema."""
    coerce_temporal_nanoseconds: bool
    """Only applicable to pandas version >= 2.0.

    A legacy option to coerce date32, date64, duration, and timestamp time units to nanoseconds when converting to pandas.
    This is the default behavior in pandas version 1.x.
    Set this option to True if you'd like to use this coercion when using pandas version >= 2.0 for backwards compatibility (not recommended otherwise).
    """


__all__ = [
    "CompliantDataFrame",
    "CompliantLazyFrame",
    "CompliantSeries",
    "DataFrameT",
    "Frame",
    "FrameT",
    "IntoDataFrame",
    "IntoDataFrameT",
    "IntoExpr",
    "IntoFrame",
    "IntoFrameT",
    "IntoSeries",
    "IntoSeriesT",
]
