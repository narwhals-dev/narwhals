from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import ExprNamespace, Function, IRNamespace
from narwhals._plan.options import FunctionOptions

if TYPE_CHECKING:
    from narwhals._plan.dummy import Expr


class ListFunction(Function, accessor="list"): ...


class Len(ListFunction, options=FunctionOptions.elementwise): ...


class IRListNamespace(IRNamespace):
    def len(self) -> Len:
        return Len()


class ExprListNamespace(ExprNamespace[IRListNamespace]):
    @property
    def _ir_namespace(self) -> type[IRListNamespace]:
        return IRListNamespace

    def len(self) -> Expr:
        return self._with_unary(self._ir.len())
