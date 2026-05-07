from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from narwhals._plan._function import Function
from narwhals._plan.expressions.namespace import ExprNamespace, IRNamespace

if TYPE_CHECKING:
    from narwhals._plan.expr import Expr


# fmt: off
class CategoricalFunction(Function, accessor="cat"): ...
class GetCategories(CategoricalFunction): ...
# fmt: on
class IRCatNamespace(IRNamespace):
    get_categories: ClassVar = GetCategories


class ExprCatNamespace(ExprNamespace[IRCatNamespace]):
    @property
    def _ir_namespace(self) -> type[IRCatNamespace]:
        return IRCatNamespace

    def get_categories(self) -> Expr:
        return self._with_unary(self._ir.get_categories())
