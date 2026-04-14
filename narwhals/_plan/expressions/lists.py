from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import narwhals._plan.dtypes_mapper as dtm
from narwhals._plan import _parameters as params
from narwhals._plan._dispatch import DispatcherOptions
from narwhals._plan._dtype import ResolveDType
from narwhals._plan._flags import FunctionFlags
from narwhals._plan._function import Function
from narwhals._plan.expressions.namespace import IRNamespace

if TYPE_CHECKING:
    from typing_extensions import Self, TypeAlias

    from narwhals._plan.expressions import FunctionExpr as FExpr
    from narwhals._plan.options import SortOptions
    from narwhals._plan.schema import FrozenSchema
    from narwhals.dtypes import DType

# NOTE: See https://github.com/astral-sh/ty/issues/1777#issuecomment-3618906859
ELEMENTWISE = FunctionFlags.ELEMENTWISE
map_first = ResolveDType.function.map_first
same_dtype = ResolveDType.function.same_dtype


# fmt: off
class ListFunction(Function, dispatch=DispatcherOptions(accessor_name="list"), flags=ELEMENTWISE): ...
class _ListInner(ListFunction):
    def resolve_dtype(self, node: FExpr[Self], schema: FrozenSchema, /) -> DType:
        return dtm.inner_dtype(node.input[0].resolve_dtype(schema), repr(self))  # pragma: no cover
class Sum(ListFunction, dtype=map_first(dtm.nested_sum_dtype)): ...
class Join(ListFunction, dtype=map_first(dtm.list_join_dtype)):
    __slots__ = ("ignore_nulls", "separator")
    separator: str
    ignore_nulls: bool
class Contains(ListFunction, dtype=dtm.BOOL):
    __function_parameters__: ClassVar[params.Binary] = params.Binary(right=params.SCALAR)
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
