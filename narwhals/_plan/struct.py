from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan.common import ExprNamespace, Function, IRNamespace
from narwhals._plan.options import FConfig, FunctionOptions

if TYPE_CHECKING:
    from narwhals._plan.dummy import Expr


class StructFunction(Function, accessor="struct"): ...


class FieldByName(
    StructFunction, options=FunctionOptions.elementwise, config=FConfig.renamed("field")
):
    __slots__ = ("name",)
    name: str

    def __repr__(self) -> str:
        return f"{super().__repr__()}({self.name!r})"


class IRStructNamespace(IRNamespace):
    def field(self, name: str) -> FieldByName:
        return FieldByName(name=name)


class ExprStructNamespace(ExprNamespace[IRStructNamespace]):
    @property
    def _ir_namespace(self) -> type[IRStructNamespace]:
        return IRStructNamespace

    def field(self, name: str) -> Expr:
        return self._with_unary(self._ir.field(name))
