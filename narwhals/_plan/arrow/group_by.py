from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._plan import expressions as ir
from narwhals._plan.arrow import acero, functions as fn
from narwhals._plan.common import temp
from narwhals._plan.expressions import aggregation as agg
from narwhals._plan.protocols import EagerDataFrameGroupBy
from narwhals._utils import Implementation

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from typing_extensions import Self, TypeAlias

    from narwhals._plan.arrow.dataframe import ArrowDataFrame as Frame
    from narwhals._plan.arrow.typing import ChunkedArray
    from narwhals._plan.expressions import NamedIR
    from narwhals._plan.typing import Seq

Incomplete: TypeAlias = Any


BACKEND_VERSION = Implementation.PYARROW._backend_version()

SUPPORTED_AGG: Mapping[type[agg.AggExpr], acero.Aggregation] = {
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
SUPPORTED_IR: Mapping[type[ir.ExprIR], acero.Aggregation] = {
    ir.Len: "hash_count_all",
    ir.Column: "hash_list",
}
SUPPORTED_FUNCTION: Mapping[type[ir.Function], acero.Aggregation] = {
    ir.boolean.All: "hash_all",
    ir.boolean.Any: "hash_any",
    ir.functions.Unique: "hash_distinct",
}

REQUIRES_PYARROW_20: tuple[Literal["kurtosis"], Literal["skew"]] = (
    "kurtosis",  # Compute the kurtosis of values in each group
    "skew",  # Compute the skewness of values in each group
)
"""They don't show in [our version of the stubs], but are possible in [`pyarrow>=20`].

[our version of the stubs]: https://github.com/narwhals-dev/narwhals/issues/2124#issuecomment-3191374210
[`pyarrow>=20`]: https://arrow.apache.org/docs/20.0/python/compute.html#grouped-aggregations
"""


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
        self.spec: acero.AggSpec

    @property
    def output_name(self) -> acero.OutputName:
        return self.named_ir.name

    def _parse_agg_expr(
        self, expr: agg.AggExpr
    ) -> tuple[acero.Target, acero.Aggregation, acero.Opts]:
        if agg_name := SUPPORTED_AGG.get(type(expr)):
            option: acero.Opts = None
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

    def _parse_function_expr(
        self, expr: ir.FunctionExpr
    ) -> tuple[acero.Target, acero.Aggregation, acero.Opts]:
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
        input_name: acero.Target = ()
        option: acero.Opts = None
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


_NULL_FILL: Final = pc.JoinOptions(
    null_handling="replace", null_replacement="__nw_null_value__"
)


def concat_str(
    native: pa.Table,
    subset: Seq[str],
    *,
    separator: str = "",
    options: pc.JoinOptions = _NULL_FILL,
) -> ChunkedArray:
    # get key columns, casting everything to str
    # docs says "list-like", runtime supports iterable
    df = native.select(subset)  # pyright: ignore[reportArgumentType]
    schema = df.schema
    dtype = (
        pa.string()
        if not any(pa.types.is_large_string(tp) for tp in schema.types)
        else pa.large_string()
    )
    schema = pa.schema((name, dtype) for name in schema.names)
    sep = fn.lit(separator, dtype)
    concat: Incomplete = pc.binary_join_element_wise
    return concat(*df.cast(schema).itercolumns(), sep, options=options)  # type: ignore[no-any-return]


class ArrowGroupBy(EagerDataFrameGroupBy["Frame"]):
    _df: Frame
    _keys: Seq[NamedIR]
    _key_names: Seq[str]
    _key_names_original: Seq[str]

    @property
    def compliant(self) -> Frame:
        return self._df

    def __iter__(self) -> Iterator[tuple[Any, Frame]]:
        temp_name = temp.column_name(self.compliant)
        temp_expr = pc.field(temp_name)
        composite_values = concat_str(self.compliant.native, self.key_names)
        re_keyed = self.compliant.native.add_column(0, temp_name, composite_values)
        from_native = self.compliant._with_native
        for v in composite_values.unique():
            # filter the keyed table to rows that have the same key (`t`)
            # then drop the temporary key on the result
            t = from_native(acero.filter_table(re_keyed, temp_expr == v).remove_column(0))
            # subset this new table to only the actual key name columns
            # then convert the first row to `tuple[pa.Scalar, ...]`
            row = t.select_names(*self.key_names).row(0)
            # convert those scalars to python literals
            group_key = tuple(el.as_py() for el in row)
            # select (all) columns from (`t`) that we started with at `<df>.group_by()``, ignoring new keys/aliases
            partition = t.select_names(*self._column_names_original)
            yield group_key, partition

    def agg(self, irs: Seq[NamedIR]) -> Frame:
        aggs: list[acero.AggSpec] = []
        use_threads: bool = True
        for e in irs:
            expr = ArrowAggExpr(e).parse()
            use_threads = use_threads and expr.use_threads
            aggs.append(expr.spec)
        native = self.compliant.native
        key_names = self.key_names
        result = self.compliant._with_native(
            acero.group_by_table(native, key_names, aggs, use_threads=use_threads)
        )
        if original := self._key_names_original:
            return result.rename(dict(zip(key_names, original)))
        return result
