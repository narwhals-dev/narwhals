from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import narwhals._plan.dtypes_mapper as dtm
from narwhals._plan._dtype import ResolveDType
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
    from narwhals._plan.schema import FrozenSchema
    from narwhals._plan.typing import IntoExpr
    from narwhals.dtypes import DType

# NOTE: See https://github.com/astral-sh/ty/issues/1777#issuecomment-3618906859
elementwise = FunctionOptions.elementwise
map_first = ResolveDType.function.map_first
same_dtype = ResolveDType.function.same_dtype


# fmt: off
class ListFunction(Function, accessor="list", options=elementwise): ...
class _ListInner(ListFunction):
    def resolve_dtype(self, schema: FrozenSchema, node: FExpr[Function]) -> DType:
        return dtm.inner_dtype(node.input[0].resolve_dtype(schema), repr(self))
class Sum(ListFunction, dtype=map_first(dtm.nested_sum_dtype)): ...
class Join(ListFunction, dtype=map_first(dtm.list_join_dtype)):
    __slots__ = ("ignore_nulls", "separator")
    separator: str
    ignore_nulls: bool
class Contains(ListFunction, dtype=dtm.BOOL):
    """N-ary (expr, item)."""

    def unwrap_input(self, node: FExpr[Self], /) -> tuple[ExprIR, ExprIR]:
        expr, item = node.input
        return expr, item
class Any(ListFunction, dtype=dtm.BOOL): ...
class All(ListFunction, dtype=dtm.BOOL): ...
class First(_ListInner): ...
class Last(_ListInner): ...
class Min(_ListInner): ...
class Max(_ListInner): ...
class Mean(ListFunction, dtype=map_first(dtm.nested_mean_median_dtype)): ...
class Median(ListFunction, dtype=map_first(dtm.nested_mean_median_dtype)): ...
class NUnique(ListFunction, dtype=dtm.IDX_DTYPE): ...
class Len(ListFunction, dtype=dtm.IDX_DTYPE): ...
class Unique(ListFunction, dtype=same_dtype()): ...
class Get(_ListInner):
    __slots__ = ("index",)
    index: int
class Sort(ListFunction, dtype=same_dtype()):
    __slots__ = ("options",)
    options: SortOptions
# fmt: on


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
