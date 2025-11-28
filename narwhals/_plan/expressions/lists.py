from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from narwhals._plan._function import Function
from narwhals._plan._parse import parse_into_expr_ir
from narwhals._plan.exceptions import function_arg_non_scalar_error
from narwhals._plan.expressions.namespace import ExprNamespace, IRNamespace
from narwhals._plan.options import FunctionOptions
from narwhals._utils import ensure_type
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan.expr import Expr
    from narwhals._plan.expressions import ExprIR, FunctionExpr as FExpr
    from narwhals._plan.typing import IntoExpr


# fmt: off
class ListFunction(Function, accessor="list", options=FunctionOptions.elementwise): ...
class Len(ListFunction): ...
class Unique(ListFunction): ...
class Get(ListFunction):
    __slots__ = ("index",)
    index: int
# fmt: on
class Contains(ListFunction):
    """N-ary (expr, item)."""

    def unwrap_input(
        self, node: FExpr[Self], /
    ) -> tuple[ExprIR, ExprIR]:  # pragma: no cover
        expr, item = node.input
        return expr, item


class IRListNamespace(IRNamespace):
    len: ClassVar = Len
    unique: ClassVar = Unique  # pragma: no cover
    contains: ClassVar = Contains

    def get(self, index: int) -> Get:
        return Get(index=index)


class ExprListNamespace(ExprNamespace[IRListNamespace]):
    @property
    def _ir_namespace(self) -> type[IRListNamespace]:
        return IRListNamespace

    def len(self) -> Expr:
        return self._with_unary(self._ir.len())

    def unique(self) -> Expr:  # pragma: no cover
        return self._with_unary(self._ir.unique())

    def get(self, index: int) -> Expr:
        ensure_type(index, int, param_name="index")
        if index < 0:
            msg = f"`index` is out of bounds; must be >= 0, got {index}"
            raise InvalidOperationError(msg)
        return self._with_unary(self._ir.get(index))

    def contains(self, item: IntoExpr) -> Expr:
        item_ir = parse_into_expr_ir(item, str_as_lit=True)
        contains = self._ir.contains()
        if not item_ir.is_scalar:
            raise function_arg_non_scalar_error(contains, "item", item_ir)
        return self._expr._from_ir(contains.to_function_expr(self._expr._ir, item_ir))
