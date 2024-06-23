"""
Public type hints.

We recommend only ever using this within a `TYPE_CHECKING` block, e.g.:

    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from narwhals.typing import DataFrame
"""

from __future__ import annotations

from typing import TYPE_CHECKING  # pragma: no cover
from typing import Literal
from typing import Union  # pragma: no cover

if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias
    from narwhals.dataframe import DataFrame
    from narwhals.dataframe import LazyFrame
    from narwhals.expression import Expr
    from narwhals.series import Series
else:
    DataFrame = object
    LazyFrame = object
    Expr = object
    Series = object

API_VERSION = Literal["0.20", "1.0"]

IntoExpr: TypeAlias = Union[Expr, str, int, float, Series]

__all__ = [
    "IntoExpr",
    "DataFrame",
    "LazyFrame",
    "Expr",
    "Series",
    "API_VERSION",
]
