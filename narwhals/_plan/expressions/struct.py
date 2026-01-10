from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from narwhals._plan._function import Function
from narwhals._plan.expressions.namespace import ExprNamespace, IRNamespace
from narwhals._plan.options import FEOptions, FunctionOptions

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan._expr_ir import ExprIR
    from narwhals._plan.expr import Expr
    from narwhals._plan.expressions.expr import StructExpr


class StructFunction(Function, accessor="struct"):
    def to_function_expr(self, *inputs: ExprIR) -> StructExpr[Self]:
        from narwhals._plan.expressions.expr import StructExpr

        return StructExpr(input=inputs, function=self, options=self.function_options)

    @property
    def needs_expansion(self) -> bool:
        msg = f"{type(self).__name__}.needs_expansion"
        raise NotImplementedError(msg)


class FieldByName(
    StructFunction, options=FunctionOptions.elementwise, config=FEOptions.renamed("field")
):
    __slots__ = ("name",)
    name: str

    def __repr__(self) -> str:
        return f"{super().__repr__()}({self.name!r})"

    @property
    def needs_expansion(self) -> bool:
        return True


class IRStructNamespace(IRNamespace):
    field: ClassVar = FieldByName


class ExprStructNamespace(ExprNamespace[IRStructNamespace]):
    @property
    def _ir_namespace(self) -> type[IRStructNamespace]:
        return IRStructNamespace

    def field(self, name: str) -> Expr:
        return self._with_unary(self._ir.field(name=name))
