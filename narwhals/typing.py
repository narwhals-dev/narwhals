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
"""DataFrame or LazyFrame"""

# TypeVars for some of the above
IntoFrameT = TypeVar("IntoFrameT", bound="IntoFrame")
IntoDataFrameT = TypeVar("IntoDataFrameT", bound="IntoDataFrame")
FrameT = TypeVar("FrameT", "DataFrame[Any]", "LazyFrame[Any]")
DataFrameT = TypeVar("DataFrameT", bound="DataFrame[Any]")

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
