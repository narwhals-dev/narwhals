from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan._immutable import Immutable
from narwhals._utils import zip_strict

if TYPE_CHECKING:
    from narwhals._plan.expressions import ExprIR, SelectorIR
    from narwhals._plan.options import ExplodeOptions, SortMultipleOptions
    from narwhals._plan.typing import Seq
    from narwhals.typing import JoinStrategy, UniqueKeepStrategy


# `DslPlan`
class LogicalPlan(Immutable): ...


class SingleInput(LogicalPlan):
    __slots__ = ("input",)
    input: LogicalPlan


class Select(SingleInput):
    __slots__ = ("exprs",)
    exprs: Seq[ExprIR]
    # NOTE: Contains a `should_broadcast` flag, but AFAICT
    # is only replaced with `False` during optimization (not when building the plan)
    # `options: ProjectionOptions`


# NOTE: Probably rename to `WithColumns`
class HStack(SingleInput):
    __slots__ = ("exprs",)
    exprs: Seq[ExprIR]
    # NOTE: Same `ProjectionOptions` comment as `Select`


class Filter(SingleInput):
    __slots__ = ("predicate",)
    predicate: ExprIR


class GroupBy(SingleInput):
    __slots__ = ("aggs", "keys")
    keys: Seq[ExprIR]
    aggs: Seq[ExprIR]


# NOTE: Probably rename to `Unique`
class Distinct(SingleInput):
    __slots__ = ("options",)
    options: DistinctOptions


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


# NOTE: Probably rename to `VConcat`
class Union(LogicalPlan):
    # `concat(how= "vertical" | "diagonal")`
    __slots__ = ("inputs", "options")
    inputs: Seq[LogicalPlan]
    options: UnionOptions


class HConcat(LogicalPlan):
    # `concat(how="horizontal")`
    __slots__ = ("inputs", "strict")
    inputs: Seq[LogicalPlan]
    strict: bool
    """Require all DataFrames to be the same height, raising an error if not, default False"""


# NOTE: `DslFunction`
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


# NOTE: Options classes (eventually move to `_plan.options`)


# NOTE: Roughly mirroring `polars`, but don't like this
# - `subset` will be (`SelectorIR` or `Seq[SelectorIR]`) | None
# - I would've had this as `Distinct.subset`, not `Distinct.options.subset`
class DistinctOptions(Immutable):
    __slots__ = ("keep", "maintain_order", "subset")
    subset: Seq[ExprIR] | None
    keep: UniqueKeepStrategy
    maintain_order: bool


class UnionOptions(Immutable):
    __slots__ = ("diagonal", "maintain_order", "to_supertypes")
    diagonal: bool
    """True for `how="diagonal"`"""

    to_supertypes: bool
    """True for [`"*_relaxed"` variants]

    [`"*_relaxed"` variants]: https://github.com/narwhals-dev/narwhals/pull/3191#issuecomment-3389117044
    """

    maintain_order: bool
    """True when using `concat`, False when using [`union`].

    [`union`]: https://github.com/pola-rs/polars/pull/24298
    """
