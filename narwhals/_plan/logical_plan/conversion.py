"""Convert `LogicalPlan` -> `ResolvedPlan`.

Intending for a rough equivalence of [`dsl_to_ir`]:

    DslPlan     -> IR
    LogicalPlan -> ResolvedPlan

[`dsl_to_ir`]: https://github.com/pola-rs/polars/blob/8f60a2d641daf7f9eeac69694b5c952f4cc34099/crates/polars-plan/src/plans/conversion/dsl_to_ir/mod.rs
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from narwhals._plan import expressions as ir
from narwhals._plan._expansion import expand_selector_irs_names, prepare_projection
from narwhals._plan.exceptions import (
    column_not_found_error,
    invalid_dtype_operation_error,
)
from narwhals._plan.logical_plan import resolved as rp
from narwhals._plan.schema import FrozenSchema, freeze_schema
from narwhals._utils import (
    Version,
    check_column_names_are_unique as raise_duplicate_error,
)
from narwhals.exceptions import ComputeError, DuplicateError, InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from typing_extensions import TypeAlias

    from narwhals._plan._expr_ir import NamedIR
    from narwhals._plan.logical_plan import plan as lp
    from narwhals._plan.typing import Seq
    from narwhals.dtypes import DType
    from narwhals.typing import Backend


Incomplete: TypeAlias = Any
"""Node is not represented in `ResolvedPlan` (yet?).

Either:
1. `polars` lowers to a simpler representation
2. `narwhals`-only node, which *may* be able to do the same
"""

dtypes = Version.MAIN.dtypes


# TODO @dangotbanned: Very big item
def expressions_to_schema(exprs: Seq[NamedIR], schema: FrozenSchema) -> FrozenSchema:
    """[`expressions_to_schema`] is a missing step at the end of `prepare_projection`.

    There's some placeholders in `FrozenSchema` ([`.select()`], [`.with_columns()`]).

    Truly understanding the `DType`s will require [#3396].

    [`expressions_to_schema`]: https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/utils.rs#L218-L245
    [`.select()`]: https://github.com/narwhals-dev/narwhals/blob/ddd93cd4b95d9760fe87cf0d7e29d87b24615777/narwhals/_plan/schema.py#L56-L68
    [`.with_columns()`]: https://github.com/narwhals-dev/narwhals/blob/ddd93cd4b95d9760fe87cf0d7e29d87b24615777/narwhals/_plan/schema.py#L73-L78
    [#3396]: https://github.com/narwhals-dev/narwhals/pull/3396
    """
    msg = "TODO: expressions_to_schema"
    raise NotImplementedError(msg)


def align_diagonal(inputs: Seq[rp.ResolvedPlan]) -> Seq[rp.ResolvedPlan]:
    """Port of `AlignDiagonal` (used for `ibis`)."""
    schemas = tuple(input.schema for input in inputs)
    it_schemas = iter(schemas)
    union = dict(next(it_schemas))
    seen = union.keys()
    for schema in it_schemas:
        union.update((nm, dtype) for nm, dtype in schema.items() if nm not in seen)
    return _align_diagonal(inputs, schemas, union)


def _align_diagonal(
    plans: Iterable[rp.ResolvedPlan],
    schemas: Iterable[FrozenSchema],
    union_schema: Mapping[str, DType],
) -> Seq[rp.ResolvedPlan]:
    output_schema = freeze_schema(union_schema)
    union_names = union_schema.keys()
    null_exprs: dict[str, NamedIR] = {}

    def iter_missing_exprs(missing: Iterable[str]) -> Iterator[NamedIR]:
        nonlocal null_exprs
        expr: NamedIR | None
        for name in missing:
            if (expr := null_exprs.get(name)) is None:
                dtype = union_schema[name]
                expr = null_exprs[name] = ir.named_ir(name, ir.lit(None, dtype))
            yield expr

    def align(plan: rp.ResolvedPlan, columns: Iterable[str]) -> rp.ResolvedPlan:
        # Even if all fields are present, we always reorder the columns to match between plans.
        if missing := union_names - columns:
            exprs = tuple(iter_missing_exprs(missing))
            msg = 'TODO: `concat(how="diagonal")` (aligned `output_schema`)'
            raise NotImplementedError(msg)
            plan = rp.WithColumns(input=plan, exprs=exprs, output_schema=...)
        return rp.SelectNames(input=plan, output_schema=output_schema)

    return tuple(align(plan, schema.keys()) for plan, schema in zip(plans, schemas))


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

    def collect(self, plan: lp.Collect, /) -> rp.Collect:
        return rp.Collect(input=self.to_resolved(plan.input))

    def concat_horizontal(self, plan: lp.HConcat, /) -> rp.HConcat:
        raise NotImplementedError

    # TODO @dangotbanned: `how="diagonal"` (Need aligned intermediate schemas for `WithColumns`)
    # Come back to after adding `select`, `with_columns`
    def concat_vertical(self, plan: lp.VConcat, /) -> rp.VConcat:
        opts = plan.options
        inputs = tuple(self.to_resolved(input) for input in plan.inputs)
        if not inputs:
            msg = "expected at least one input in 'concat'"
            raise InvalidOperationError(msg)
        if opts.diagonal:
            inputs = align_diagonal(inputs)
        if opts.to_supertypes:
            msg = (
                f"TODO: `{type(opts).__name__}.to_supertypes`\nRequires:\n"
                "- https://github.com/narwhals-dev/narwhals/pull/3396\n"
                "- https://github.com/narwhals-dev/narwhals/issues/3386"
            )
            raise NotImplementedError(msg)
        output_schema = inputs[0].schema
        for other in inputs[1:]:
            if (schema := other.schema) != output_schema:
                msg = f"'concat' inputs should all have the same schema, got\n{dict(output_schema)} and \n{dict(schema)}"
                raise InvalidOperationError(msg)
        return rp.VConcat(inputs=inputs, maintain_order=opts.maintain_order)

    def explode(self, plan: lp.MapFunction[lp.Explode], /) -> rp.ResolvedPlan:
        input = self.to_resolved(plan.input)
        input_schema = input.schema
        f_explode = plan.function
        columns = expand_selector_irs_names(
            (f_explode.columns,), schema=input_schema, require_any=True
        )
        allowed = dtypes.List, dtypes.Array
        schema = dict(input_schema)
        for to_explode in columns:
            dtype = schema[to_explode]
            if not isinstance(dtype, allowed):
                raise invalid_dtype_operation_error(dtype, "explode", *allowed)
            inner = dtype.inner
            schema[to_explode] = inner if not isinstance(inner, type) else inner()
        return rp.MapFunction(
            input=input,
            function=rp.Explode(
                columns=columns,
                options=f_explode.options,
                output_schema=freeze_schema(schema),
            ),
        )

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

    def map_function(self, plan: lp.MapFunction[lp.LpFunctionT_co], /) -> rp.ResolvedPlan:
        return plan.function.resolve(self, plan)

    def pivot(self, plan: lp.Pivot, /) -> Incomplete:
        raise NotImplementedError

    def rename(self, plan: lp.MapFunction[lp.Rename], /) -> rp.ResolvedPlan:
        input = self.to_resolved(plan.input)
        f_rename = plan.function
        if not f_rename.old:
            return input
        input_schema = input.schema
        before = set(f_rename.old)
        after = f_rename.new
        names = deque[str]()
        exprs = deque[ir.NamedIR]()
        for idx, name in enumerate(input_schema):
            if name in before:
                actual = after[idx]
                before.remove(name)
            else:
                actual = name
            exprs.append(ir.named_ir(actual, ir.col(name)))
            names.append(actual)

        if before:
            # we had extra names not present in the schema
            raise column_not_found_error(f_rename.old, input.schema)
        return rp.Select(
            input=input,
            exprs=tuple(exprs),
            output_schema=freeze_schema(zip(names, input_schema.values())),
        )

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
        input = self.to_resolved(plan.input)
        return rp.SelectNames(
            input=input, output_schema=input.schema.select_names(plan.names)
        )

    def sink_parquet(self, plan: lp.SinkParquet, /) -> rp.SinkParquet:
        return rp.SinkParquet(input=self.to_resolved(plan.input), target=plan.target)

    def slice(self, plan: lp.Slice, /) -> rp.Slice:
        return rp.Slice(
            input=self.to_resolved(plan.input), offset=plan.offset, length=plan.length
        )

    def sort(self, plan: lp.Sort, /) -> rp.Sort:
        input = self.to_resolved(plan.input)
        by = expand_selector_irs_names(plan.by, schema=input.schema, require_any=True)
        opts = plan.options
        n_by = len(by)
        n_desc = len(opts.descending)
        if n_desc == 1:
            desc = opts.descending * n_by
        elif n_desc == n_by:
            desc = opts.descending
        else:
            msg = f"the length of `descending` ({n_desc}) does not match the length of `by` ({n_by})"
            raise ComputeError(msg)
        if len(opts.nulls_last) not in {1, n_by}:
            msg = f"the length of `descending` ({len(opts.nulls_last)}) does not match the length of `by` ({n_by})"
            raise ComputeError(msg)
        # NOTE: `polars` expands `nulls_last` here too, but `pyarrow<=23` doesn't support per-key
        # See https://github.com/apache/arrow/pull/46926
        return rp.Sort(input=input, by=by, options=opts.__replace__(descending=desc))

    def unique(self, plan: lp.Unique, /) -> rp.Unique:
        input = self.to_resolved(plan.input)
        subset: Seq[str] | None = None
        if s_irs := plan.subset:
            schema = input.schema
            subset = expand_selector_irs_names(s_irs, schema=schema, require_any=True)
        return rp.Unique(input=input, subset=subset, options=plan.options)

    def unique_by(self, plan: lp.UniqueBy, /) -> Incomplete:
        raise NotImplementedError

    def unnest(self, plan: lp.MapFunction[lp.Unnest], /) -> rp.MapFunction[rp.Unnest]:
        # NOTE: Pretty delicate process to optimize for
        input = self.to_resolved(plan.input)
        input_schema = input.schema
        columns = expand_selector_irs_names(
            (plan.function.columns,), schema=input_schema, require_any=True
        )
        tp_struct = dtypes.Struct

        # We need to insert struct fields in the same order as the original, while removing the struct itself
        schema: dict[str, DType] = {}

        # guard against duplicates from unnesting
        # allows skipping the duplicate check on iter 1
        seen = schema.keys()

        # struct columns that will be replaced with their fields
        # used destructively to allow skipping everything after the last target struct is handled
        to_unnest = set(columns)

        for name, dtype in input_schema.items():
            if not to_unnest or name not in to_unnest:
                schema[name] = dtype
            elif not isinstance(dtype, tp_struct):
                raise invalid_dtype_operation_error(dtype, "unnest", tp_struct)
            elif not seen or seen.isdisjoint(f.name for f in dtype.fields):
                schema.update(
                    (f.name, (f.dtype if not isinstance(f.dtype, type) else f.dtype()))
                    for f in dtype.fields
                )
                to_unnest.remove(name)
            else:
                raise_duplicate_error((*seen, *(f.name for f in dtype.fields)))
        return rp.MapFunction(
            input=input,
            function=rp.Unnest(columns=columns, output_schema=freeze_schema(schema)),
        )

    def unpivot(self, plan: lp.MapFunction[lp.Unpivot], /) -> rp.MapFunction[rp.Unpivot]:
        raise NotImplementedError

    def with_columns(self, plan: lp.WithColumns, /) -> rp.WithColumns:
        raise NotImplementedError

    # TODO @dangotbanned: Unify `DType` to either:
    # - UInt32 (polars excluding `bigidx`)
    # - UInt64 (pyarrow in some cases)
    # - Int64 (most backends)
    def with_row_index(self, plan: lp.MapFunction[lp.RowIndex], /) -> rp.ResolvedPlan:
        input = self.to_resolved(plan.input)
        input_schema = input.schema
        name = plan.function.name
        if name in input_schema:
            msg = f"Duplicate column name {name!r}"
            raise DuplicateError(msg)
        output_schema = freeze_schema({name: dtypes.Int64()} | input_schema._mapping)
        return rp.MapFunction(
            input=input, function=rp.RowIndex(name=name, output_schema=output_schema)
        )

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
