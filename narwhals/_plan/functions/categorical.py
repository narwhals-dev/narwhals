from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.expressions.categorical import IRCatNamespace
from narwhals._plan.expressions.namespace import ExprNamespace

if TYPE_CHECKING:
    from narwhals._plan.expr import Expr


class ExprCatNamespace(ExprNamespace[IRCatNamespace]):
    @property
    def _ir_namespace(self) -> type[IRCatNamespace]:
        return IRCatNamespace

    def get_categories(self) -> Expr:
        return self._with_unary(self._ir.get_categories())
