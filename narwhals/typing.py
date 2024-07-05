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
    from narwhals.expression import Expr
    from narwhals.series import Series

    # All dataframes supported by Narwhals have a
    # `columns` property. Their similarities don't extend
    # _that_ much further unfortunately...
    class NativeDataFrame(Protocol):
        @property
        def columns(self) -> Any: ...

        def join(self, *args: Any, **kwargs: Any) -> Any: ...


# Anything which can be converted to an expression.
IntoExpr: TypeAlias = Union["Expr", str, int, float, "Series"]
# Anything which can be converted to a Narwhals DataFrame.
IntoDataFrame: TypeAlias = Union["NativeDataFrame", "DataFrame[Any]"]
IntoDataFrameT = TypeVar("IntoDataFrameT", bound="IntoDataFrame")

__all__ = ["IntoExpr", "IntoDataFrame"]
