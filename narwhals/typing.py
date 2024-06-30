from __future__ import annotations

from typing import TYPE_CHECKING  # pragma: no cover
from typing import Any
from typing import Protocol
from typing import Union  # pragma: no cover

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
    # `columns` property.
    class NativeDataFrame(Protocol):
        @property
        def columns(self) -> Any: ...


IntoExpr: TypeAlias = Union["Expr", str, int, float, "Series"]
IntoDataFrame: TypeAlias = Union["NativeDataFrame", "DataFrame"]

__all__ = ["IntoExpr", "IntoDataFrame"]
