from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import Protocol
from typing import TypeVar
from typing import Union

from narwhals._compliant import CompliantDataFrame
from narwhals._compliant import CompliantLazyFrame
from narwhals._compliant import CompliantSeries

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Iterable
    from typing import Sized

    import numpy as np
    from typing_extensions import TypeAlias

    from narwhals import dtypes
    from narwhals.dataframe import DataFrame
    from narwhals.dataframe import LazyFrame
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

    # TODO @dangotbanned: `nw.Series` **cannot** be allowed to match this!!!
    class NativeSeries(Sized, Iterable[Any], Protocol):
        def filter(self, *args: Any, **kwargs: Any) -> Any: ...

    class DataFrameLike(Protocol):
        def __dataframe__(self, *args: Any, **kwargs: Any) -> Any: ...


class SupportsNativeNamespace(Protocol):
    def __native_namespace__(self) -> ModuleType: ...


IntoExpr: TypeAlias = Union["Expr", str, "Series[Any]"]
"""Anything which can be converted to an expression.

Use this to mean "either a Narwhals expression, or something which can be converted
into one". For example, `exprs` in `DataFrame.select` is typed to accept `IntoExpr`,
as it can either accept a `nw.Expr` (e.g. `df.select(nw.col('a'))`) or a string
which will be interpreted as a `nw.Expr`, e.g. `df.select('a')`.
"""

IntoDataFrame: TypeAlias = Union["NativeFrame", "DataFrame[Any]", "DataFrameLike"]
"""Anything which can be converted to a Narwhals DataFrame.

Use this if your function accepts a narwhalifiable object but doesn't care about its backend.

Examples:
    >>> import narwhals as nw
    >>> from narwhals.typing import IntoDataFrame
    >>> def agnostic_shape(df_native: IntoDataFrame) -> tuple[int, int]:
    ...     df = nw.from_native(df_native, eager_only=True)
    ...     return df.shape
"""

IntoLazyFrame: TypeAlias = "NativeLazyFrame | LazyFrame[Any]"

IntoFrame: TypeAlias = Union[
    "NativeFrame", "DataFrame[Any]", "LazyFrame[Any]", "DataFrameLike"
]
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

FrameT = TypeVar("FrameT", bound="Frame")
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

_ShapeT = TypeVar("_ShapeT", bound="tuple[int, ...]")
_NDArray: TypeAlias = "np.ndarray[_ShapeT, Any]"
_1DArray: TypeAlias = "_NDArray[tuple[int]]"  # noqa: PYI042
_2DArray: TypeAlias = "_NDArray[tuple[int, int]]"  # noqa: PYI042, PYI047
_AnyDArray: TypeAlias = "_NDArray[tuple[int, ...]]"  # noqa: PYI047
_NumpyScalar: TypeAlias = "np.generic[Any]"
Into1DArray: TypeAlias = "_1DArray | _NumpyScalar"
"""A 1-dimensional `numpy.ndarray` or scalar that can be converted into one."""


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
