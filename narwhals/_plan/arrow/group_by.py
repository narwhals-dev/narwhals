from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pyarrow as pa  # ignore-banned-import
import pyarrow.acero as pac
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._plan import expressions as ir
from narwhals._plan.expressions import aggregation as agg
from narwhals._plan.protocols import DataFrameGroupBy
from narwhals._utils import Implementation, requires

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from typing_extensions import Self, TypeAlias

    from narwhals._arrow.typing import (  # type: ignore[attr-defined]
        AggregateOptions,
        Aggregation,
    )
    from narwhals._plan.arrow.dataframe import ArrowDataFrame
    from narwhals._plan.expressions import NamedIR
    from narwhals._plan.typing import Seq

Incomplete: TypeAlias = Any

AceroTarget: TypeAlias = "tuple[()] | list[str]"
NativeAggSpec: TypeAlias = "tuple[AceroTarget, Aggregation, AggregateOptions | None]"
OutputName: TypeAlias = str
AceroAggSpec: TypeAlias = (
    "tuple[AceroTarget, Aggregation, AggregateOptions | None, OutputName]"
)


BACKEND_VERSION = Implementation.PYARROW._backend_version()

SUPPORTED_AGG: Mapping[type[agg.AggExpr], Aggregation] = {
    agg.Sum: "hash_sum",
    agg.Mean: "hash_mean",
    agg.Median: "hash_approximate_median",
    agg.Max: "hash_max",
    agg.Min: "hash_min",
    agg.Std: "hash_stddev",
    agg.Var: "hash_variance",
    agg.Count: "hash_count",
    agg.Len: "hash_count",
    agg.NUnique: "hash_count_distinct",
    agg.First: "hash_first",
    agg.Last: "hash_last",
}
SUPPORTED_IR: Mapping[type[ir.ExprIR], Aggregation] = {
    ir.Len: "hash_count_all",
    ir.Column: "hash_list",
}
SUPPORTED_FUNCTION: Mapping[type[ir.Function], Aggregation] = {
    ir.boolean.All: "hash_all",
    ir.boolean.Any: "hash_any",
    ir.functions.Unique: "hash_distinct",
}

REMAINING: tuple[Aggregation, ...] = (
    "hash_first_last",  # Compute the first and last of values in each group
    "hash_min_max",  # Compute the minimum and maximum of values in each group
    "hash_one",  # Get one value from each group
    "hash_product",  # Compute the product of values in each group
    "hash_tdigest",  # Compute approximate quantiles of values in each group
)
"""Available [native aggs] we haven't used.

[native aggs]: https://arrow.apache.org/docs/python/compute.html#grouped-aggregations
"""


REQUIRES_PYARROW_20: tuple[
    Literal["kurtosis"], Literal["pivot_wider"], Literal["skew"]
] = (
    "kurtosis",  # Compute the kurtosis of values in each group
    "pivot_wider",  # Pivot values according to a pivot key column
    "skew",  # Compute the skewness of values in each group
)
"""https://arrow.apache.org/docs/20.0/python/compute.html#grouped-aggregations"""


# TODO @dangotbanned: Factor out to just a `bool.__ior__`
def _ensure_single_thread(
    grouped: pa.TableGroupBy, expr: ir.OrderableAggExpr, /
) -> pa.TableGroupBy:
    """First/last require disabling threading."""
    if BACKEND_VERSION >= (14, 0) and grouped._use_threads:
        # NOTE: Stubs say `_table` is a method, but at runtime it is a property
        grouped = pa.TableGroupBy(grouped._table, grouped.keys, use_threads=False)  # type: ignore[arg-type]
    elif BACKEND_VERSION < (14, 0):  # pragma: no cover
        msg = (
            f"Using `{expr!r}` in a `group_by().agg(...)` context is only available in 'pyarrow>=14.0.0', "
            f"found version {requires._unparse_version(BACKEND_VERSION)!r}.\n\n"
            f"See https://github.com/apache/arrow/issues/36709"
        )
        raise NotImplementedError(msg)
    return grouped


def group_by_error(
    expr: ArrowAggExpr,
    reason: Literal[
        "too complex",
        "unsupported aggregation",
        "unsupported function",
        "unsupported expression",
    ],
) -> NotImplementedError:
    if reason == "too complex":
        msg = "Non-trivial complex aggregation found"
    else:
        msg = reason.title()
    msg = f"{msg} in 'pyarrow.Table':\n\n{expr.named_ir!r}"
    return NotImplementedError(msg)


class ArrowAggExpr:
    def __init__(self, named_ir: NamedIR, /) -> None:
        self.named_ir: NamedIR = named_ir

    @property
    def output_name(self) -> OutputName:
        return self.named_ir.name

    def _parse_agg_expr(
        self, expr: agg.AggExpr, grouped: pa.TableGroupBy
    ) -> tuple[AceroTarget, Aggregation, AggregateOptions | None, pa.TableGroupBy]:
        if agg_name := SUPPORTED_AGG.get(type(expr)):
            option: AggregateOptions | None = None
            if isinstance(expr, (agg.Std, agg.Var)):
                # NOTE: Only branch which needs an instance (for `ddof`)
                option = pc.VarianceOptions(ddof=expr.ddof)
            elif isinstance(expr, (agg.NUnique, agg.Len)):
                option = pc.CountOptions(mode="all")
            elif isinstance(expr, agg.Count):
                option = pc.CountOptions(mode="only_valid")
            elif isinstance(expr, (agg.First, agg.Last)):
                option = pc.ScalarAggregateOptions(skip_nulls=False)
                # NOTE: Only branch which needs access to `pa.TableGroupBy`
                grouped = _ensure_single_thread(grouped, expr)
            if isinstance(expr.expr, ir.Column):
                return [expr.expr.name], agg_name, option, grouped
            raise group_by_error(self, "too complex")
        raise group_by_error(self, "unsupported aggregation")

    def _parse_function_expr(self, expr: ir.FunctionExpr) -> NativeAggSpec:
        func = expr.function
        if agg_name := SUPPORTED_FUNCTION.get(type(func)):
            if isinstance(func, (ir.boolean.All, ir.boolean.Any)):
                option = pc.ScalarAggregateOptions(min_count=0)
            else:
                option = None
        else:
            raise group_by_error(self, "unsupported function")
        if len(expr.input) == 1 and isinstance(expr.input[0], ir.Column):
            return [expr.input[0].name], agg_name, option
        raise group_by_error(self, "too complex")

    def to_native(self, grouped: pa.TableGroupBy) -> tuple[pa.TableGroupBy, AceroAggSpec]:
        expr = self.named_ir.expr
        input_name: AceroTarget = ()
        option: AggregateOptions | None = None
        if isinstance(expr, agg.AggExpr):
            input_name, agg_name, option, grouped = self._parse_agg_expr(expr, grouped)
        elif isinstance(expr, ir.FunctionExpr):
            input_name, agg_name, option = self._parse_function_expr(expr)
        elif isinstance(expr, (ir.Len, ir.Column)):
            agg_name = SUPPORTED_IR[type(expr)]
            if isinstance(expr, ir.Column):
                input_name = [expr.name]
        else:
            raise group_by_error(self, "unsupported expression")
        agg_spec = input_name, agg_name, option, self.output_name
        return grouped, agg_spec


class ArrowGroupBy(DataFrameGroupBy["ArrowDataFrame"]):
    _df: ArrowDataFrame
    _grouped: pa.TableGroupBy
    _keys: Seq[NamedIR]
    _keys_names: Seq[str]

    @classmethod
    def by_names(cls, df: ArrowDataFrame, names: Seq[str], /) -> Self:
        obj = cls.__new__(cls)
        obj._df = df
        obj._keys = ()
        obj._keys_names = names
        obj._grouped = pa.TableGroupBy(df.native, list(names))
        return obj

    @classmethod
    def by_named_irs(cls, df: ArrowDataFrame, irs: Seq[NamedIR], /) -> Self:
        raise NotImplementedError

    @property
    def compliant(self) -> ArrowDataFrame:
        return self._df

    def __iter__(self) -> Iterator[tuple[Any, ArrowDataFrame]]:
        raise NotImplementedError

    def agg(self, irs: Seq[NamedIR]) -> ArrowDataFrame:
        gb = self._grouped
        aggs: list[AceroAggSpec] = []
        for e in irs:
            gb, agg_spec = ArrowAggExpr(e).to_native(gb)
            aggs.append(agg_spec)
        result = _aggregate(
            self.compliant.native,
            list(self.keys_names),
            aggs,
            use_threads=gb._use_threads,
        )
        return self.compliant._with_native(result)


def _aggregate(
    df: pa.Table,
    keys: list[str],
    aggregations: Iterable[  # TODO @dangotbanned: Revisit after replacing `_ensure_single_thread`
        AceroAggSpec
    ],
    *,
    use_threads: bool,
) -> pa.Table:
    """Adapted from [`pa.TableGroupBy.aggregate`](https://github.com/apache/arrow/blob/0e7e70cfdef4efa287495272649c071a700c34fa/python/pyarrow/table.pxi#L6600-L6626)."""
    aggs = list(aggregations) if not isinstance(aggregations, list) else aggregations
    return _group_by(df, keys, aggs, use_threads=use_threads)


def _group_by(
    table: pa.Table,
    keys: list[str],
    aggregates: list[AceroAggSpec],
    *,
    use_threads: bool = True,
) -> pa.Table:
    """Backport of [apache/arrow#36768].

    `first` and `last` were [broken in `pyarrow==13`].

    Also allows us to specify our own aliases for aggregate output columns.

    [apache/arrow#36768]: https://github.com/apache/arrow/pull/36768
    [broken in `pyarrow==13`]: https://github.com/apache/arrow/issues/36709
    """
    # NOTE: Stubs are (incorrectly) invariant
    aggs: Incomplete = aggregates
    keys_: Incomplete = keys
    decls = [
        pac.Declaration("table_source", pac.TableSourceNodeOptions(table)),
        pac.Declaration("aggregate", pac.AggregateNodeOptions(aggs, keys=keys_)),
    ]
    return pac.Declaration.from_sequence(decls).to_table(use_threads=use_threads)
