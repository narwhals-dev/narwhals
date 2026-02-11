"""Convert `LogicalPlan` -> `ResolvedPlan`.

Intending for a rough equivalence of [`dsl_to_ir`]:

    DslPlan     -> IR
    LogicalPlan -> ResolvedPlan

[`dsl_to_ir`]: https://github.com/pola-rs/polars/blob/8f60a2d641daf7f9eeac69694b5c952f4cc34099/crates/polars-plan/src/plans/conversion/dsl_to_ir/mod.rs
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._plan.logical_plan import plan as lp, resolved as rp

if TYPE_CHECKING:
    from typing_extensions import TypeAlias, TypeIs

    from narwhals._plan.schema import FrozenSchema


Incomplete: TypeAlias = Any


def is_scan(
    plan: lp.LogicalPlan,
) -> TypeIs[lp.ScanDataFrame | lp.ScanCsv | lp.ScanParquet | lp.ScanParquetImpl[Any]]:
    return not plan.has_inputs


# TODO @dangotbanned: Plan how schema resolution should work
def schema(plan: lp.LogicalPlan) -> FrozenSchema:
    msg = f"TODO: schema({type(plan).__name__})"
    raise NotImplementedError(msg)


# - `to_alp` is called with empty (`expr_arena`, `lp_arena`) for the initial plan
#  - https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-lazy/src/frame/cached_arenas.rs#L114
# - `to_alp_impl` is the recursive part
#  - each traversal passes through the (mutable) `DslConversionContext`
#    - https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/conversion/dsl_to_ir/utils.rs#L4-L14
#  - (not doing this part) that leads to `ConversionOptimizer` https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/conversion/stack_opt.rs
class Resolver:
    """Lower the `LogicalPlan` into `ResolvedPlan`.

    - Case branches in [`dsl_to_ir::to_alp_impl`] should correspond to methods here
      - Backends can override translations by subclassing
    - Likely need some outer context
      - Most of these will be recursive

    [`polars_plan::plans::conversion::dsl_to_ir::to_alp_impl`]: https://github.com/pola-rs/polars/blob/8f60a2d641daf7f9eeac69694b5c952f4cc34099/crates/polars-plan/src/plans/conversion/dsl_to_ir/mod.rs#L142-L1666
    """

    # TODO @dangotbanned: All transforming nodes
    def convert(self, plan: lp.LogicalPlan) -> rp.ResolvedPlan:
        # lets do the "dumb" version first
        if is_scan(plan):
            if isinstance(plan, lp.ScanDataFrame):
                return self.scan_dataframe(plan)
            if isinstance(plan, lp.ScanParquet):
                if isinstance(plan, lp.ScanParquetImpl):
                    return self.scan_parquet_impl(plan)
                return self.scan_parquet(plan)
            return self.scan_csv(plan)
        if isinstance(plan, (lp.Collect, lp.SinkParquet)):
            if isinstance(plan, lp.Collect):
                return self.collect(plan)
            return self.sink_parquet(plan)

        msg = f"TODO: Resolver.convert({type(plan).__name__})"
        raise NotImplementedError(msg)

    # Scan
    def scan_dataframe(
        self, plan: lp.ScanDataFrame, *args: Any, **kwds: Any
    ) -> rp.ScanDataFrame:
        raise NotImplementedError

    def scan_csv(self, plan: lp.ScanCsv, *args: Any, **kwds: Any) -> rp.ScanCsv:
        raise NotImplementedError

    def scan_parquet(
        self, plan: lp.ScanParquet, *args: Any, **kwds: Any
    ) -> rp.ScanParquet:
        raise NotImplementedError

    # NOTE: Experimental!
    def scan_parquet_impl(
        self, plan: lp.ScanParquetImpl[Any], *args: Any, **kwds: Any
    ) -> rp.ScanParquet:
        raise NotImplementedError

    # Sink
    def collect(self, plan: lp.Collect, *args: Any, **kwds: Any) -> rp.Collect:
        raise NotImplementedError

    def sink_parquet(
        self, plan: lp.SinkParquet, *args: Any, **kwds: Any
    ) -> rp.SinkParquet:
        raise NotImplementedError

    # Everything else
    def concat_horizontal(self, plan: lp.HConcat, *args: Any, **kwds: Any) -> rp.HConcat:
        raise NotImplementedError

    def concat_vertical(self, plan: lp.HConcat, *args: Any, **kwds: Any) -> rp.VConcat:
        raise NotImplementedError

    def explode(
        self, plan: lp.MapFunction[lp.Explode], *args: Any, **kwds: Any
    ) -> rp.MapFunction[rp.Explode]:
        raise NotImplementedError

    def filter(self, plan: lp.Filter, *args: Any, **kwds: Any) -> rp.Filter:
        raise NotImplementedError

    def group_by(self, plan: lp.GroupBy, *args: Any, **kwds: Any) -> rp.GroupBy:
        raise NotImplementedError

    def join(self, plan: lp.Join, *args: Any, **kwds: Any) -> rp.Join:
        raise NotImplementedError

    def join_asof(self, plan: lp.JoinAsof, *args: Any, **kwds: Any) -> rp.JoinAsof:
        raise NotImplementedError

    def map_function(
        self, plan: lp.MapFunction, *args: Any, **kwds: Any
    ) -> rp.MapFunction:
        raise NotImplementedError

    def pivot(self, plan: lp.Pivot, *args: Any, **kwds: Any) -> Incomplete:
        raise NotImplementedError

    def rename(
        self, plan: lp.MapFunction[lp.Rename], *args: Any, **kwds: Any
    ) -> Incomplete:
        raise NotImplementedError

    def select(self, plan: lp.Select, *args: Any, **kwds: Any) -> rp.Select:
        raise NotImplementedError

    def select_names(
        self, plan: lp.SelectNames, *args: Any, **kwds: Any
    ) -> rp.SelectNames:
        raise NotImplementedError

    def slice(self, plan: lp.Slice, *args: Any, **kwds: Any) -> rp.Slice:
        raise NotImplementedError

    def sort(self, plan: lp.Sort, *args: Any, **kwds: Any) -> rp.Sort:
        raise NotImplementedError

    def unique(self, plan: lp.Unique, *args: Any, **kwds: Any) -> rp.Unique:
        raise NotImplementedError

    def unique_by(self, plan: lp.UniqueBy, *args: Any, **kwds: Any) -> Incomplete:
        raise NotImplementedError

    def unnest(
        self, plan: lp.MapFunction[lp.Unnest], *args: Any, **kwds: Any
    ) -> rp.MapFunction[rp.Unnest]:
        raise NotImplementedError

    def unpivot(
        self, plan: lp.MapFunction[lp.Unpivot], *args: Any, **kwds: Any
    ) -> rp.MapFunction[rp.Unpivot]:
        raise NotImplementedError

    def with_columns(
        self, plan: lp.WithColumns, *args: Any, **kwds: Any
    ) -> rp.WithColumns:
        raise NotImplementedError

    def with_row_index(
        self, plan: lp.MapFunction[lp.RowIndex], *args: Any, **kwds: Any
    ) -> rp.MapFunction[rp.RowIndex]:
        raise NotImplementedError

    def with_row_index_by(
        self, plan: lp.MapFunction[lp.RowIndexBy], *args: Any, **kwds: Any
    ) -> Incomplete:
        raise NotImplementedError
