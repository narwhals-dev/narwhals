from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan._immutable import Immutable

if TYPE_CHECKING:
    from narwhals._plan.expressions import ExprIR
    from narwhals._plan.options import SortMultipleOptions
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


class RowIndex(LpFunction): ...


class Explode(LpFunction): ...


class Unnest(LpFunction): ...


class Rename(LpFunction): ...
