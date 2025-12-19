from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan._immutable import Immutable
from narwhals._utils import zip_strict

if TYPE_CHECKING:
    from narwhals._plan.expressions import ExprIR, SelectorIR
    from narwhals._plan.options import ExplodeOptions, SortMultipleOptions
    from narwhals._plan.typing import Seq
    from narwhals.typing import JoinStrategy


# `DslPlan`
class LogicalPlan(Immutable): ...


class SingleInput(LogicalPlan):
    __slots__ = ("input",)
    input: LogicalPlan


class Select(SingleInput):
    __slots__ = ("exprs",)
    exprs: Seq[ExprIR]
    # options: ProjectionOptions  # noqa: ERA001 (contains a `should_broadcast` flag)


class Filter(SingleInput):
    predicate: ExprIR


class GroupBy(SingleInput):
    __slots__ = ("aggs", "keys")
    keys: Seq[ExprIR]
    aggs: Seq[ExprIR]


class HStack(SingleInput):
    __slots__ = ("exprs",)
    # `with_columns`
    exprs: Seq[ExprIR]


class Distinct(SingleInput):
    # `unique`
    # options: DistinctOptions  # noqa: ERA001 (subset, keep, maintain_order)
    ...


class Sort(SingleInput):
    __slots__ = ("by", "options")
    by: Seq[ExprIR]
    options: SortMultipleOptions


class Slice(SingleInput):
    __slots__ = ("length", "offset")
    offset: int
    length: int | None


class Join(LogicalPlan):
    __slots__ = ("how", "input_left", "input_right", "left_on", "right_on", "suffix")
    # NOTE: Might be nicer to have `inputs: tuple[LogicalPlan, LogicalPlan]`
    input_left: LogicalPlan
    input_right: LogicalPlan
    left_on: Seq[str]
    right_on: Seq[str]
    how: JoinStrategy
    suffix: str


class Union(LogicalPlan):
    __slots__ = ("inputs",)
    # `concat(how= "vertical" | "diagonal")`
    inputs: Seq[LogicalPlan]
    # args: UnionArgs  # noqa: ERA001


class HConcat(LogicalPlan):
    __slots__ = ("inputs",)
    # `concat(how="horizontal")`
    inputs: Seq[LogicalPlan]
    # options: HConcatOptions  # noqa: ERA001


# `DslFunction`
class LpFunction(Immutable): ...


class Explode(LpFunction):
    __slots__ = ("columns", "options")
    columns: SelectorIR
    options: ExplodeOptions


class Unnest(LpFunction):
    __slots__ = ("columns",)
    columns: SelectorIR


class RowIndex(LpFunction):
    __slots__ = ("name",)
    name: str


class Rename(LpFunction):
    __slots__ = ("new", "old")
    old: Seq[str]
    new: Seq[str]

    @property
    def mapping(self) -> dict[str, str]:
        # Trying to avoid adding mutable fields
        return dict(zip_strict(self.old, self.new))
