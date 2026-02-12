"""Convert `LogicalPlan` -> `ResolvedPlan`.

Intending for a rough equivalence of [`dsl_to_ir`]:

    DslPlan     -> IR
    LogicalPlan -> ResolvedPlan

[`dsl_to_ir`]: https://github.com/pola-rs/polars/blob/8f60a2d641daf7f9eeac69694b5c952f4cc34099/crates/polars-plan/src/plans/conversion/dsl_to_ir/mod.rs
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from narwhals._plan._expansion import prepare_projection
from narwhals._plan.logical_plan import resolved as rp
from narwhals._plan.schema import freeze_schema
from narwhals.exceptions import ComputeError

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals._plan.logical_plan import plan as lp
    from narwhals.typing import Backend


Incomplete: TypeAlias = Any
"""Node is not represented in `ResolvedPlan` (yet?).

Either:
1. `polars` lowers to a simpler representation
2. `narwhals`-only node, which *may* be able to do the same
"""


# - `to_alp` is called with empty (`expr_arena`, `lp_arena`) for the initial plan
#  - https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-lazy/src/frame/cached_arenas.rs#L114
# - `to_alp_impl` is the recursive part
#  - each traversal passes through the (mutable) `DslConversionContext`
#    - https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/conversion/dsl_to_ir/utils.rs#L4-L14
#  - (not doing this part) that leads to `ConversionOptimizer` https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/conversion/stack_opt.rs
class Resolver:
    """Default conversion from `LogicalPlan` into `ResolvedPlan`.

    - Case branches in [`dsl_to_ir::to_alp_impl`] should correspond to methods here
      - Backends can override translations by subclassing
    - Likely need some outer context
      - Most of these will be recursive

    [`polars_plan::plans::conversion::dsl_to_ir::to_alp_impl`]: https://github.com/pola-rs/polars/blob/8f60a2d641daf7f9eeac69694b5c952f4cc34099/crates/polars-plan/src/plans/conversion/dsl_to_ir/mod.rs#L142-L1666
    """

    __slots__ = ()

    def to_resolved(self, plan: lp.LogicalPlan, /) -> rp.ResolvedPlan:
        """Converts `LogicalPlan` to `ResolvedPlan`.

        All implementations should call this on the inputs of each plan as the first step.
        """
        return plan.resolve(self)

    # TODO @dangotbanned: Implement everything
    def collect(self, plan: lp.Collect, /) -> rp.Collect:
        raise NotImplementedError

    def concat_horizontal(self, plan: lp.HConcat, /) -> rp.HConcat:
        raise NotImplementedError

    def concat_vertical(self, plan: lp.VConcat, /) -> rp.VConcat:
        raise NotImplementedError

    def explode(self, plan: lp.MapFunction[lp.Explode], /) -> rp.MapFunction[rp.Explode]:
        raise NotImplementedError

    def filter(self, plan: lp.Filter, /) -> rp.Filter:
        input = self.to_resolved(plan.input)
        named_irs, _ = prepare_projection((plan.predicate,), schema=input.schema)
        if len(named_irs) != 1:
            msg = (
                f"The predicate passed to 'LazyFrame.filter' expanded to {len(named_irs)!r} expressions:\n\n{named_irs!r}\n"
                "This is ambiguous. Try to combine the predicates with the 'all' or `any' expression."
            )
            raise ComputeError(msg)
        return rp.Filter(input=input, predicate=named_irs[0])

    def group_by(self, plan: lp.GroupBy, /) -> rp.GroupBy:
        raise NotImplementedError

    def join(self, plan: lp.Join, /) -> rp.Join:
        raise NotImplementedError

    def join_asof(self, plan: lp.JoinAsof, /) -> rp.JoinAsof:
        raise NotImplementedError

    def map_function(self, plan: lp.MapFunction[lp.LpFunctionT_co], /) -> rp.MapFunction:
        raise NotImplementedError

    def pivot(self, plan: lp.Pivot, /) -> Incomplete:
        raise NotImplementedError

    def rename(self, plan: lp.MapFunction[lp.Rename], /) -> Incomplete:
        raise NotImplementedError

    def scan_csv(self, plan: lp.ScanCsv, /) -> rp.ScanCsv:
        raise NotImplementedError

    def scan_dataframe(self, plan: lp.ScanDataFrame, /) -> rp.ScanDataFrame:
        return rp.ScanDataFrame(df=plan.df, output_schema=plan.schema)

    def scan_parquet(self, plan: lp.ScanParquet, /) -> rp.ScanParquet:
        """TODO: Need a way of getting `backend: IntoBackend` from an outer context."""
        raise NotImplementedError

    def scan_parquet_impl(self, plan: lp.ScanParquetImpl[lp.ImplT], /) -> rp.ScanParquet:
        return _scan_parquet(plan.source, plan.implementation)

    def select(self, plan: lp.Select, /) -> rp.Select:
        raise NotImplementedError

    def select_names(self, plan: lp.SelectNames, /) -> rp.SelectNames:
        raise NotImplementedError

    def sink_parquet(self, plan: lp.SinkParquet, /) -> rp.SinkParquet:
        raise NotImplementedError

    def slice(self, plan: lp.Slice, /) -> rp.Slice:
        return rp.Slice(
            input=self.to_resolved(plan.input), offset=plan.offset, length=plan.length
        )

    def sort(self, plan: lp.Sort, /) -> rp.Sort:
        raise NotImplementedError

    def unique(self, plan: lp.Unique, /) -> rp.Unique:
        raise NotImplementedError

    def unique_by(self, plan: lp.UniqueBy, /) -> Incomplete:
        raise NotImplementedError

    def unnest(self, plan: lp.MapFunction[lp.Unnest], /) -> rp.MapFunction[rp.Unnest]:
        raise NotImplementedError

    def unpivot(self, plan: lp.MapFunction[lp.Unpivot], /) -> rp.MapFunction[rp.Unpivot]:
        raise NotImplementedError

    def with_columns(self, plan: lp.WithColumns, /) -> rp.WithColumns:
        raise NotImplementedError

    def with_row_index(
        self, plan: lp.MapFunction[lp.RowIndex], /
    ) -> rp.MapFunction[rp.RowIndex]:
        raise NotImplementedError

    def with_row_index_by(self, plan: lp.MapFunction[lp.RowIndexBy], /) -> Incomplete:
        raise NotImplementedError


@lru_cache(maxsize=64)
def _scan_parquet(source: str, implementation: Backend, /) -> rp.ScanParquet:
    """Cached conversion using `read_parquet_schema`.

    ## Warning
    Very naive approach *for now* to make some progress.

    Real thing needs to cache on file metadata like:

        Path(source).resolve().stat()

    - `str` could be a relative path, and another call changes working directory
    - we could have correct file, but a stale schema in the cache
    - probably 99 other concerns
    """
    from narwhals._plan import functions as F

    schema = F.read_parquet_schema(source, backend=implementation)
    return rp.ScanParquet(source=source, output_schema=freeze_schema(schema))
