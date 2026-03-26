from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan._parse import parse_into_expr_ir
from narwhals._plan.exceptions import function_arg_non_scalar_error
from narwhals._plan.expressions.lists import IRListNamespace
from narwhals._plan.expressions.namespace import ExprNamespace
from narwhals._plan.options import SortOptions
from narwhals._utils import ensure_type
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from narwhals._plan.expr import Expr
    from narwhals._plan.typing import IntoExpr


class ExprListNamespace(ExprNamespace[IRListNamespace]):
    @property
    def _ir_namespace(self) -> type[IRListNamespace]:
        return IRListNamespace

    def min(self) -> Expr:
        return self._with_unary(self._ir.min())

    def max(self) -> Expr:
        return self._with_unary(self._ir.max())

    def mean(self) -> Expr:
        return self._with_unary(self._ir.mean())

    def median(self) -> Expr:
        return self._with_unary(self._ir.median())

    def sum(self) -> Expr:
        return self._with_unary(self._ir.sum())

    def len(self) -> Expr:
        return self._with_unary(self._ir.len())

    def unique(self) -> Expr:
        return self._with_unary(self._ir.unique())

    def get(self, index: int) -> Expr:
        ensure_type(index, int, param_name="index")
        if index < 0:
            msg = f"`index` is out of bounds; must be >= 0, got {index}"
            raise InvalidOperationError(msg)
        return self._with_unary(self._ir.get(index=index))

    def contains(self, item: IntoExpr) -> Expr:
        item_ir = parse_into_expr_ir(item, str_as_lit=True)
        contains = self._ir.contains()
        if not item_ir.is_scalar():
            raise function_arg_non_scalar_error(contains, "item", item_ir)
        return self._expr._from_ir(contains.to_function_expr(self._expr._ir, item_ir))

    def join(self, separator: str, *, ignore_nulls: bool = True) -> Expr:
        ensure_type(separator, str, param_name="separator")
        return self._with_unary(
            self._ir.join(separator=separator, ignore_nulls=ignore_nulls)
        )

    def any(self) -> Expr:
        return self._with_unary(self._ir.any())

    def all(self) -> Expr:
        return self._with_unary(self._ir.all())

    def first(self) -> Expr:
        return self._with_unary(self._ir.first())

    def last(self) -> Expr:
        return self._with_unary(self._ir.last())

    def n_unique(self) -> Expr:
        return self._with_unary(self._ir.n_unique())

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Expr:
        options = SortOptions(descending=descending, nulls_last=nulls_last)
        return self._with_unary(self._ir.sort(options=options))
