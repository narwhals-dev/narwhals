from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.expressions.namespace import ExprNamespace
from narwhals._plan.expressions.struct import IRStructNamespace

if TYPE_CHECKING:
    from narwhals._plan.expr import Expr


class ExprStructNamespace(ExprNamespace[IRStructNamespace]):
    @property
    def _ir_namespace(self) -> type[IRStructNamespace]:
        return IRStructNamespace

    def field(self, name: str) -> Expr:
        return self._with_unary(self._ir.field(name=name))
