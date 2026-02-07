from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from narwhals._plan.expressions import selectors as s_ir
from narwhals._plan.expressions.boolean import all_horizontal
from narwhals._plan.logical_plan import plan as lp
from narwhals._utils import normalize_path
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Self, TypeAlias

    from narwhals._plan.dataframe import DataFrame
    from narwhals._plan.expressions import ExprIR, SelectorIR
    from narwhals._plan.logical_plan.plan import LogicalPlan, LpFunction
    from narwhals._plan.options import (
        ExplodeOptions,
        JoinOptions,
        SortMultipleOptions,
        UniqueOptions,
        UnpivotOptions,
    )
    from narwhals._plan.typing import Seq
    from narwhals.typing import FileSource, PivotAgg


__all__ = ["LpBuilder"]

Incomplete: TypeAlias = Any


# `DslBuilder`
class LpBuilder:
    __slots__ = ("_plan",)
    _plan: LogicalPlan

    __hash__: ClassVar[None] = None  # type: ignore[assignment]
    """This class is a helper for building `LogicalPlan`s, which *are* hashable.

    Compare and operate on plans, **not the sugar around them**.
    """

    @classmethod
    def from_df(cls, df: DataFrame[Any, Any], /) -> Self:
        return cls.from_plan(lp.DataFrameScan.from_narwhals(df))

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
        return self.from_plan(lp.Select(input=self._plan, exprs=exprs))

    def with_columns(self, exprs: Seq[ExprIR], options: Incomplete = None) -> Self:
        return self.from_plan(lp.WithColumns(input=self._plan, exprs=exprs))

    def filter(self, predicate: ExprIR) -> Self:
        return self.from_plan(lp.Filter(input=self._plan, predicate=predicate))

    def group_by(self, keys: Seq[ExprIR], aggs: Seq[ExprIR]) -> Self:
        return self.from_plan(lp.GroupBy(input=self._plan, keys=keys, aggs=aggs))

    def sort(self, by: Seq[SelectorIR], options: SortMultipleOptions) -> Self:
        return self.from_plan(lp.Sort(input=self._plan, by=by, options=options))

    def join(
        self,
        other: LogicalPlan,
        left_on: Seq[str],
        right_on: Seq[str],
        options: JoinOptions,
    ) -> Self:
        return self.from_plan(
            lp.Join(
                inputs=(self._plan, other),
                left_on=left_on,
                right_on=right_on,
                options=options,
            )
        )

    def slice(self, offset: int, length: int | None = None) -> Self:
        return self.from_plan(lp.Slice(input=self._plan, offset=offset, length=length))

    def unique(self, subset: Seq[SelectorIR] | None, options: UniqueOptions) -> Self:
        return self.from_plan(lp.Unique(input=self._plan, subset=subset, options=options))

    def pivot(
        self,
        on: SelectorIR,
        on_columns: Incomplete,
        index: SelectorIR,
        values: SelectorIR,
        agg: PivotAgg | None,
        separator: str,
    ) -> Self:
        return self.from_plan(
            lp.Pivot(
                input=self._plan,
                on=on,
                on_columns=on_columns,
                index=index,
                values=values,
                agg=agg,
                separator=separator,
            )
        )

    # Terminal
    def sink(self, sink: lp.Sink) -> Self:
        if isinstance(self._plan, lp.Sink):
            msg = "cannot create a sink on top of another sink"
            raise InvalidOperationError(msg)
        return self.from_plan(sink)

    def collect(self) -> Self:
        return self.sink(lp.Collect(input=self._plan))

    def sink_parquet(self, target: FileSource) -> Self:
        return self.sink(lp.SinkParquet(input=self._plan, target=normalize_path(target)))

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
        return self.from_plan(lp.MapFunction(input=self._plan, function=function))

    # `MapFunction`
    def explode(self, columns: SelectorIR, options: ExplodeOptions) -> Self:
        return self.map(lp.Explode(columns=columns, options=options))

    def unnest(self, columns: SelectorIR) -> Self:
        return self.map(lp.Unnest(columns=columns))

    def rename(self, mapping: Mapping[str, str]) -> Self:
        return self.map(lp.Rename(old=tuple(mapping), new=tuple(mapping.values())))

    def with_row_index(self, name: str = "index") -> Self:
        return self.map(lp.RowIndex(name=name))

    def unpivot(
        self, on: SelectorIR | None, index: SelectorIR, options: UnpivotOptions
    ) -> Self:
        return self.map(lp.Unpivot(on=on, index=index, options=options))
