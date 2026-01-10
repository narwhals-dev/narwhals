from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from narwhals._plan._function import Function
from narwhals._plan._parse import parse_into_expr_ir
from narwhals._plan.exceptions import function_arg_non_scalar_error
from narwhals._plan.expressions.namespace import ExprNamespace, IRNamespace
from narwhals._plan.options import FunctionOptions, SortOptions
from narwhals._utils import ensure_type
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from typing_extensions import Self, TypeAlias

    from narwhals._plan.expr import Expr
    from narwhals._plan.expressions import ExprIR, FunctionExpr as FExpr
    from narwhals._plan.typing import IntoExpr


# fmt: off
class ListFunction(Function, accessor="list", options=FunctionOptions.elementwise): ...
class Any(ListFunction): ...
class All(ListFunction): ...
class First(ListFunction): ...
class Last(ListFunction): ...
class Min(ListFunction): ...
class Max(ListFunction): ...
class Mean(ListFunction): ...
class Median(ListFunction): ...
class NUnique(ListFunction): ...
class Sum(ListFunction): ...
class Len(ListFunction): ...
class Unique(ListFunction): ...
class Get(ListFunction):
    __slots__ = ("index",)
    index: int
class Sort(ListFunction):
    __slots__ = ("options",)
    options: SortOptions
class Join(ListFunction):
    """Join all string items in a sublist and place a separator between them."""

    __slots__ = ("ignore_nulls", "separator")
    separator: str
    ignore_nulls: bool
# fmt: on
class Contains(ListFunction):
    """N-ary (expr, item)."""

    def unwrap_input(self, node: FExpr[Self], /) -> tuple[ExprIR, ExprIR]:
        expr, item = node.input
        return expr, item


Aggregation: TypeAlias = (
    "Any | All | First | Last | Min | Max | Mean | Median | NUnique | Sum"
)


class IRListNamespace(IRNamespace):
    len: ClassVar = Len
    unique: ClassVar = Unique
    contains: ClassVar = Contains
    get: ClassVar = Get
    join: ClassVar = Join
    min: ClassVar = Min
    max: ClassVar = Max
    mean: ClassVar = Mean
    median: ClassVar = Median
    sum: ClassVar = Sum
    any: ClassVar = Any
    all: ClassVar = All
    first: ClassVar = First
    last: ClassVar = Last
    n_unique: ClassVar = NUnique
    sort: ClassVar = Sort


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
        if not item_ir.is_scalar:
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
