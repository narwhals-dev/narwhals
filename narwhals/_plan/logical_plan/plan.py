from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from narwhals._plan._immutable import Immutable
from narwhals._plan.expressions import selectors as s_ir
from narwhals._plan.expressions.boolean import all_horizontal
from narwhals._plan.schema import freeze_schema
from narwhals._utils import zip_strict

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from typing_extensions import Self, TypeAlias

    from narwhals._plan.dataframe import DataFrame
    from narwhals._plan.expressions import ExprIR, SelectorIR
    from narwhals._plan.options import (
        ExplodeOptions,
        JoinOptions,
        SortMultipleOptions,
        UniqueOptions,
        VConcatOptions,
    )
    from narwhals._plan.schema import FrozenSchema
    from narwhals._plan.typing import Seq

Incomplete: TypeAlias = Any


# `DslPlan`
# TODO @dangotbanned: Add `LogicalPlan`s for ops in `nw.*Frame`, that aren't yet in `nwp.*Frame`
class LogicalPlan(Immutable): ...


# TODO @dangotbanned: Careful think about how (non-scan) source nodes should work
# - Schema only?
# - Different for eager vs lazy?
class DataFrameScan(LogicalPlan):
    __slots__ = ("df", "schema")
    df: DataFrame
    schema: FrozenSchema

    # NOTE: Probably want a `staticmethod`, change if nothing is needed from `cls`
    @classmethod
    def from_narwhals(cls, df: DataFrame) -> DataFrameScan:
        obj = cls.__new__(cls)
        object.__setattr__(obj, "df", df.clone())
        object.__setattr__(obj, "schema", freeze_schema(df.schema))
        return obj

    @property
    def __immutable_values__(self) -> Iterator[Any]:
        # NOTE: Deferring how to handle the hash *for now*
        # Currently, every `DataFrameSource` will have a unique psuedo-hash
        # Caching a native table seems like a non-starter, once `pandas` enters the party
        yield from (id(self.df), self.schema)


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
    """Require all `inputs` to be the same height, raising an error if not, default False."""


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


# `DslBuilder`
class LpBuilder:
    __slots__ = ("_plan",)
    _plan: LogicalPlan

    __hash__: ClassVar[None] = None  # type: ignore[assignment]
    """This class is a helper for building `LogicalPlan`s, which *are* hashable.

    Compare and operate on plans, **not the sugar around them**.
    """

    @classmethod
    def from_df(cls, df: DataFrame, /) -> Self:
        return cls.from_plan(DataFrameScan.from_narwhals(df))

    @classmethod
    def from_plan(cls, plan: LogicalPlan, /) -> Self:
        obj = cls.__new__(cls)
        obj._plan = plan
        return obj

    def to_plan(self) -> LogicalPlan:
        return self._plan

    # TODO @dangotbanned: Decide on if `ProjectionOptions` should be added
    # Either replace `Incomplete` or remove `options` (and the placeholder in `fill_null`)
    def select(self, exprs: Seq[ExprIR], options: Incomplete = None) -> Self:
        return self.from_plan(Select(input=self._plan, exprs=exprs))

    def with_columns(self, exprs: Seq[ExprIR], options: Incomplete = None) -> Self:
        return self.from_plan(WithColumns(input=self._plan, exprs=exprs))

    def filter(self, predicate: ExprIR) -> Self:
        return self.from_plan(Filter(input=self._plan, predicate=predicate))

    def group_by(self, keys: Seq[ExprIR], aggs: Seq[ExprIR]) -> Self:
        return self.from_plan(GroupBy(input=self._plan, keys=keys, aggs=aggs))

    def sort(self, by: Seq[SelectorIR], options: SortMultipleOptions) -> Self:
        return self.from_plan(Sort(input=self._plan, by=by, options=options))

    def join(
        self,
        other: LogicalPlan,
        left_on: Seq[str],
        right_on: Seq[str],
        options: JoinOptions,
    ) -> Self:
        return self.from_plan(
            Join(
                input_left=self._plan,
                input_right=other,
                left_on=left_on,
                right_on=right_on,
                options=options,
            )
        )

    def slice(self, offset: int, length: int | None = None) -> Self:
        return self.from_plan(Slice(input=self._plan, offset=offset, length=length))

    def unique(self, subset: Seq[SelectorIR] | None, options: UniqueOptions) -> Self:
        return self.from_plan(Unique(input=self._plan, subset=subset, options=options))

    # Sugar
    def drop(self, columns: SelectorIR) -> Self:
        return self.select(((~columns.to_narwhals())._ir,))

    def fill_null(self, fill_value: ExprIR) -> Self:
        return self.select(
            (s_ir.all().to_narwhals().fill_null(fill_value.to_narwhals())._ir,),
            options={"duplicate_check": False},  # ProjectionOptions
        )

    # https://github.com/pola-rs/polars/blob/1684cc09dfaa46656dfecc45ab866d01aa69bc78/crates/polars-plan/src/dsl/builder_dsl.rs#L172-L175
    def drop_nulls(self, subset: SelectorIR | None) -> Self:
        predicate = all_horizontal((subset or s_ir.all()).to_narwhals().is_not_null()._ir)
        return self.filter(predicate)

    def with_column(self, expr: ExprIR) -> Self:
        return self.with_columns((expr,))

    # `DslBuilder.map_private`
    def map(self, function: LpFunction) -> Self:
        return self.from_plan(MapFunction(input=self._plan, function=function))

    # `MapFunction`
    def explode(self, columns: SelectorIR, options: ExplodeOptions) -> Self:
        return self.map(Explode(columns=columns, options=options))

    def unnest(self, columns: SelectorIR) -> Self:
        return self.map(Unnest(columns=columns))

    def rename(self, mapping: Mapping[str, str]) -> Self:
        return self.map(Rename(old=tuple(mapping), new=tuple(mapping.values())))

    def with_row_index(self, name: str = "index") -> Self:
        return self.map(RowIndex(name=name))
