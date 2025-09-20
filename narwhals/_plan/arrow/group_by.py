from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pyarrow as pa  # ignore-banned-import
import pyarrow.acero as pac
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._plan import expressions as ir
from narwhals._plan.expressions import aggregation as agg
from narwhals._plan.protocols import DataFrameGroupBy
from narwhals._utils import Implementation

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

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
        self.use_threads: bool = True
        """See https://github.com/apache/arrow/issues/36709"""
        self.spec: AceroAggSpec

    @property
    def output_name(self) -> OutputName:
        return self.named_ir.name

    def _parse_agg_expr(
        self, expr: agg.AggExpr
    ) -> tuple[AceroTarget, Aggregation, AggregateOptions | None]:
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
                self.use_threads = False
            if isinstance(expr.expr, ir.Column):
                return [expr.expr.name], agg_name, option
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

    def parse(self) -> Self:
        expr = self.named_ir.expr
        input_name: AceroTarget = ()
        option: AggregateOptions | None = None
        if isinstance(expr, agg.AggExpr):
            input_name, agg_name, option = self._parse_agg_expr(expr)
        elif isinstance(expr, ir.FunctionExpr):
            input_name, agg_name, option = self._parse_function_expr(expr)
        elif isinstance(expr, (ir.Len, ir.Column)):
            agg_name = SUPPORTED_IR[type(expr)]
            if isinstance(expr, ir.Column):
                input_name = [expr.name]
        else:
            raise group_by_error(self, "unsupported expression")
        self.spec = input_name, agg_name, option, self.output_name
        return self


class ArrowGroupBy(DataFrameGroupBy["ArrowDataFrame"]):
    _df: ArrowDataFrame
    _keys: Seq[NamedIR]
    _keys_names: Seq[str]

    @classmethod
    def by_names(
        cls, df: ArrowDataFrame, names: Seq[str], /, *, drop_null_keys: bool = False
    ) -> Self:
        obj = cls.__new__(cls)
        if drop_null_keys:
            df = df.drop_nulls(names)
        obj._df = df
        obj._keys = ()
        obj._keys_names = names
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
        aggs: list[AceroAggSpec] = []
        use_threads: bool = True
        for e in irs:
            expr = ArrowAggExpr(e).parse()
            use_threads = use_threads and expr.use_threads
            aggs.append(expr.spec)
        return self.compliant._with_native(self._agg(aggs, use_threads=use_threads))

    def _agg(self, agg_specs: list[AceroAggSpec], /, *, use_threads: bool) -> pa.Table:
        """Adapted from [`pa.TableGroupBy.aggregate`] and [`pa.acero._group_by`].

        - Backport of [apache/arrow#36768].
          - `first` and `last` were [broken in `pyarrow==13`].
        - Also allows us to specify our own aliases for aggregate output columns.
          - Fixes [narwhals-dev/narwhals#1612]

        [`pa.TableGroupBy.aggregate`]: https://github.com/apache/arrow/blob/0e7e70cfdef4efa287495272649c071a700c34fa/python/pyarrow/table.pxi#L6600-L6626
        [`pa.acero._group_by`]: https://github.com/apache/arrow/blob/0e7e70cfdef4efa287495272649c071a700c34fa/python/pyarrow/acero.py#L412-L418
        [apache/arrow#36768]: https://github.com/apache/arrow/pull/36768
        [broken in `pyarrow==13`]: https://github.com/apache/arrow/issues/36709
        [narwhals-dev/narwhals#1612]: https://github.com/narwhals-dev/narwhals/issues/1612
        """
        df = self.compliant.native
        # NOTE: Stubs are (incorrectly) invariant
        keys: Incomplete = list(self.keys_names)
        aggs: Incomplete = agg_specs
        decls = [
            pac.Declaration("table_source", pac.TableSourceNodeOptions(df)),
            pac.Declaration("aggregate", pac.AggregateNodeOptions(aggs, keys=keys)),
        ]
        return pac.Declaration.from_sequence(decls).to_table(use_threads=use_threads)
