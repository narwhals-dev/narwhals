from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from narwhals._plan._immutable import Immutable

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan._function import Unary
    from narwhals._plan.expr import Expr
    from narwhals._plan.expressions import ExprIR


class IRNamespace(Immutable):
    __slots__ = ("_ir",)
    _ir: ExprIR

    @classmethod
    def from_expr(cls, expr: Expr, /) -> Self:
        return cls(_ir=expr._ir)


IRNamespaceT = TypeVar("IRNamespaceT", bound=IRNamespace)


class ExprNamespace(Immutable, Generic[IRNamespaceT]):
    __slots__ = ("_expr",)
    _expr: Expr

    @property
    def _ir_namespace(self) -> type[IRNamespaceT]:
        raise NotImplementedError

    @property
    def _ir(self) -> IRNamespaceT:
        return self._ir_namespace.from_expr(self._expr)

    def _to_narwhals(self, ir: ExprIR, /) -> Expr:
        return self._expr._from_ir(ir)

    def _with_unary(self, function: Unary, /) -> Expr:
        return self._expr._with_unary(function)
