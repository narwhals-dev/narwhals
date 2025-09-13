from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from narwhals._plan._function import Function
from narwhals._plan.common import ExprNamespace, IRNamespace
from narwhals._plan.options import FunctionOptions

if TYPE_CHECKING:
    from narwhals._plan.expr import Expr


# fmt: off
class ListFunction(Function, accessor="list"): ...
class Len(ListFunction, options=FunctionOptions.elementwise): ...
# fmt: on
class IRListNamespace(IRNamespace):
    len: ClassVar = Len


class ExprListNamespace(ExprNamespace[IRListNamespace]):
    @property
    def _ir_namespace(self) -> type[IRListNamespace]:
        return IRListNamespace

    def len(self) -> Expr:
        return self._with_unary(self._ir.len())
