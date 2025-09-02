from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import ExprNamespace, Function, IRNamespace

if TYPE_CHECKING:
    from narwhals._plan.dummy import Expr


class CategoricalFunction(Function, accessor="cat"): ...


class GetCategories(CategoricalFunction): ...


class IRCatNamespace(IRNamespace):
    def get_categories(self) -> GetCategories:
        return GetCategories()


class ExprCatNamespace(ExprNamespace[IRCatNamespace]):
    @property
    def _ir_namespace(self) -> type[IRCatNamespace]:
        return IRCatNamespace

    def get_categories(self) -> Expr:
        return self._with_unary(self._ir.get_categories())
