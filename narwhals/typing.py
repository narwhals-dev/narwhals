from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol
from typing import TypeVar
from typing import Union

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
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

    class DataFrameLike(Protocol):
        def __dataframe__(self, *args: Any, **kwargs: Any) -> Any: ...


IntoExpr: TypeAlias = Union["Expr", str, "Series"]
"""Anything which can be converted to an expression."""

IntoDataFrame: TypeAlias = Union["NativeFrame", "DataFrame[Any]", "DataFrameLike"]
"""Anything which can be converted to a Narwhals DataFrame."""

IntoFrame: TypeAlias = Union[
    "NativeFrame", "DataFrame[Any]", "LazyFrame[Any]", "DataFrameLike"
]
"""Anything which can be converted to a Narwhals DataFrame or LazyFrame."""

Frame: TypeAlias = Union["DataFrame[Any]", "LazyFrame[Any]"]
"""Narwhals DataFrame or Narwhals LazyFrame"""

# TypeVars for some of the above
IntoFrameT = TypeVar("IntoFrameT", bound="IntoFrame")
IntoDataFrameT = TypeVar("IntoDataFrameT", bound="IntoDataFrame")
FrameT = TypeVar("FrameT", bound="Frame")
DataFrameT = TypeVar("DataFrameT", bound="DataFrame[Any]")


class DTypes:
    Int64: type[dtypes.Int64]
    Int32: type[dtypes.Int32]
    Int16: type[dtypes.Int16]
    Int8: type[dtypes.Int8]
    UInt64: type[dtypes.UInt64]
    UInt32: type[dtypes.UInt32]
    UInt16: type[dtypes.UInt16]
    UInt8: type[dtypes.UInt8]
    Float64: type[dtypes.Float64]
    Float32: type[dtypes.Float32]
    String: type[dtypes.String]
    Boolean: type[dtypes.Boolean]
    Object: type[dtypes.Object]
    Categorical: type[dtypes.Categorical]
    Enum: type[dtypes.Enum]
    Datetime: type[dtypes.Datetime]
    Duration: type[dtypes.Duration]
    Date: type[dtypes.Date]
    Struct: type[dtypes.Struct]
    List: type[dtypes.List]
    Array: type[dtypes.Array]
    Unknown: type[dtypes.Unknown]


__all__ = [
    "IntoExpr",
    "IntoDataFrame",
    "IntoDataFrameT",
    "IntoFrame",
    "IntoFrameT",
    "Frame",
    "FrameT",
    "DataFrameT",
]
