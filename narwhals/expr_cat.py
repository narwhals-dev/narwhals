from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Generic
from typing import TypeVar

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.expr import Expr

ExprT = TypeVar("ExprT", bound="Expr")


class ExprCatNamespace(Generic[ExprT]):
    def __init__(self: Self, expr: ExprT) -> None:
        self._expr = expr

    def get_categories(self: Self) -> ExprT:
        """Get unique categories from column.

        Returns:
            A new expression.
        """
        return self._expr.__class__(
            lambda plx: self._expr._to_compliant_expr(plx).cat.get_categories()
        )
