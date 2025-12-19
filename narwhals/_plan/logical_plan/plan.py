from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from narwhals._plan._immutable import Immutable
from narwhals._utils import zip_strict

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Self, TypeAlias

    from narwhals._plan.expressions import ExprIR, SelectorIR
    from narwhals._plan.options import ExplodeOptions, SortMultipleOptions
    from narwhals._plan.typing import Seq
    from narwhals.typing import JoinStrategy, UniqueKeepStrategy

Incomplete: TypeAlias = Any


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


# `DslPlan::HStack`
class WithColumns(SingleInput):
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


# `DslPlan::Distinct`
class Unique(SingleInput):
    __slots__ = ("options", "subset")
    subset: Seq[SelectorIR] | None
    options: UniqueOptions


class Sort(SingleInput):
    __slots__ = ("by", "options")
    by: Seq[SelectorIR]
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
    options: JoinOptions


class MapFunction(SingleInput):
    # `polars` says this is for UDFs, but uses it for: `Rename`, `RowIndex`, `Unnest`, `Explode`
    __slots__ = ("function",)
    function: LpFunction


# `DslPlan::Union`
class VConcat(LogicalPlan):
    """`concat(how= "vertical" | "diagonal")`."""

    __slots__ = ("inputs", "options")
    inputs: Seq[LogicalPlan]
    options: VConcatOptions


class HConcat(LogicalPlan):
    """`concat(how="horizontal")`."""

    __slots__ = ("inputs", "strict")
    inputs: Seq[LogicalPlan]
    strict: bool
    """Require all `inputs` to be the same height, raising an error if not, default False"""


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


class UniqueOptions(Immutable):
    __slots__ = ("keep", "maintain_order")
    keep: UniqueKeepStrategy
    maintain_order: bool


class VConcatOptions(Immutable):
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


class JoinOptions(Immutable):
    __slots__ = ("how", "suffix")
    how: JoinStrategy
    suffix: str


# NOTE: Using a `Protocol` to allow empty body(s)
# Just trying to scope things out
class LpBuilder(Protocol):
    def project(self, exprs: Seq[ExprIR], options: Incomplete) -> Self: ...
    def with_columns(self, exprs: Seq[ExprIR], options: Incomplete) -> Self: ...
    def filter(self, predicate: ExprIR) -> Self: ...
    def group_by(self, keys: Seq[ExprIR], aggs: Seq[ExprIR]) -> Self: ...
    def sort(self, by: Seq[SelectorIR], options: SortMultipleOptions) -> Self: ...
    def join(
        self,
        other: LogicalPlan,
        left_on: Seq[str],
        right_on: Seq[str],
        options: JoinOptions,
    ) -> Self: ...
    def slice(self, offset: int, length: int | None = None) -> Self: ...
    def unique(self, subset: Seq[SelectorIR] | None, options: UniqueOptions) -> Self: ...

    # Sugar
    def drop(self, columns: SelectorIR) -> Self: ...
    def fill_null(
        self, fill_value: ExprIR
    ) -> Self: ...  # ProjectionOptions {duplicate_check: false}
    # This has a pretty cool impl https://github.com/pola-rs/polars/blob/1684cc09dfaa46656dfecc45ab866d01aa69bc78/crates/polars-plan/src/dsl/builder_dsl.rs#L172-L175
    def drop_nulls(self, subset: SelectorIR | None) -> Self: ...
    def with_column(self, expr: ExprIR) -> Self: ...

    # `MapFunction`
    def explode(self, columns: SelectorIR, options: ExplodeOptions) -> Self: ...
    def unnest(self, columns: SelectorIR) -> Self: ...
    def rename(self, mapping: Mapping[str, str]) -> Self: ...
    def with_row_index(self, name: str = "index") -> Self: ...

    @classmethod
    def from_plan(cls, plan: LogicalPlan, /) -> Self: ...
    def to_plan(self) -> LogicalPlan: ...
    def with_plan(self, plan: LogicalPlan, /) -> Self: ...
