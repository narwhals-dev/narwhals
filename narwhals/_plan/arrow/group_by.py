from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._compliant.typing import NarwhalsAggregation as _NarwhalsAggregation
from narwhals._plan import expressions as ir
from narwhals._plan.expressions import aggregation as agg
from narwhals._plan.protocols import DataFrameGroupBy
from narwhals._utils import Implementation, requires

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


NarwhalsAggregation: TypeAlias = Literal[_NarwhalsAggregation, "first", "last"]
InputName: TypeAlias = "str | tuple[()]"
"""`()` can be used with `"count_all"`."""

NativeName: TypeAlias = str
OutputName: TypeAlias = str
NativeAggSpec: TypeAlias = "tuple[InputName, Aggregation, AggregateOptions | None]"
RenameSpec: TypeAlias = tuple[NativeName, OutputName]


BACKEND_VERSION = Implementation.PYARROW._backend_version()


# TODO @dangotbanned: Missing `nw.col("a").len()`
SUPPORTED_AGG: Mapping[type[agg.AggExpr], Aggregation] = {
    agg.Sum: "sum",
    agg.Mean: "mean",
    agg.Median: "approximate_median",
    agg.Max: "max",
    agg.Min: "min",
    agg.Std: "stddev",
    agg.Var: "variance",
    agg.Count: "count",
    agg.Len: "count",
    agg.NUnique: "count_distinct",
    agg.First: "first",
    agg.Last: "last",
}


SUPPORTED_IR: Mapping[type[ir.Len], Aggregation] = {ir.Len: "count_all"}
SUPPORTED_FUNCTION: Mapping[type[ir.boolean.BooleanFunction], Aggregation] = {
    ir.boolean.All: "all",
    ir.boolean.Any: "any",
}

REMAINING: tuple[Aggregation, ...] = (
    "distinct",  # Keep the distinct values in each group
    "first_last",  # Compute the first and last of values in each group
    "list",  # List all values in each group
    "min_max",  # Compute the minimum and maximum of values in each group
    "one",  # Get one value from each group
    "product",  # Compute the product of values in each group
    "tdigest",  # Compute approximate quantiles of values in each group
)
"""Available [native aggs] we haven't used (excluding `first`, `last`)

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
    ) -> tuple[InputName, Aggregation, AggregateOptions | None, pa.TableGroupBy]:
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
                return expr.expr.name, agg_name, option, grouped
            raise group_by_error(self, "too complex")
        raise group_by_error(self, "unsupported aggregation")

    def _parse_function_expr(self, expr: ir.FunctionExpr) -> NativeAggSpec:
        if isinstance(expr.function, (ir.boolean.All, ir.boolean.Any)):
            agg_name = SUPPORTED_FUNCTION[type(expr.function)]
            option = pc.ScalarAggregateOptions(min_count=0)
            if len(expr.input) == 1 and isinstance(expr.input[0], ir.Column):
                return expr.input[0].name, agg_name, option
            raise group_by_error(self, "too complex")
        raise group_by_error(self, "unsupported function")

    def _rename_spec(self, input_name: InputName, agg_name: Aggregation, /) -> RenameSpec:
        # `pyarrow` auto-generates the lhs
        # we want to overwrite that later with rhs
        old = f"{input_name}_{agg_name}" if input_name else agg_name
        return old, self.output_name

    def to_native(
        self, grouped: pa.TableGroupBy
    ) -> tuple[pa.TableGroupBy, NativeAggSpec, RenameSpec]:
        expr = self.named_ir.expr
        if isinstance(expr, agg.AggExpr):
            input_name, agg_name, option, grouped = self._parse_agg_expr(expr, grouped)
        elif isinstance(expr, ir.Len):
            input_name, agg_name, option = ((), "count_all", None)
        elif isinstance(expr, ir.Column):
            input_name, agg_name, option = (expr.name, "list", None)
        elif isinstance(expr, ir.FunctionExpr):
            input_name, agg_name, option = self._parse_function_expr(expr)
        else:
            raise group_by_error(self, "unsupported expression")
        agg_spec = input_name, agg_name, option
        return grouped, agg_spec, self._rename_spec(input_name, agg_name)


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
        aggs: list[NativeAggSpec] = []
        renames: list[RenameSpec] = []
        for e in irs:
            gb, agg_spec, rename = ArrowAggExpr(e).to_native(gb)
            aggs.append(agg_spec)
            renames.append(rename)
        return self.compliant._with_native(gb.aggregate(aggs)).rename(dict(renames))
