from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._plan import expressions as ir
from narwhals._plan._dispatch import get_dispatch_name
from narwhals._plan._guards import is_agg_expr, is_function_expr
from narwhals._plan.arrow import acero, functions as fn, options
from narwhals._plan.compliant.group_by import EagerDataFrameGroupBy
from narwhals._plan.expressions import aggregation as agg
from narwhals._utils import Implementation
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from typing_extensions import Self, TypeAlias

    from narwhals._plan.arrow.dataframe import ArrowDataFrame as Frame
    from narwhals._plan.arrow.typing import ChunkedArray
    from narwhals._plan.expressions import NamedIR
    from narwhals._plan.typing import Seq

Incomplete: TypeAlias = Any

# NOTE: Unless stated otherwise, all aggregations have 2 variants:
# - `<function>` (pc.Function.kind == "scalar_aggregate")
# - `hash_<function>` (pc.Function.kind == "hash_aggregate")
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
    ir.Column: "hash_list",  # `hash_aggregate` only
}
SUPPORTED_FUNCTION: Mapping[type[ir.Function], acero.Aggregation] = {
    ir.boolean.All: "hash_all",
    ir.boolean.Any: "hash_any",
    ir.functions.Unique: "hash_distinct",  # `hash_aggregate` only
}

REQUIRES_PYARROW_20: tuple[Literal["kurtosis"], Literal["skew"]] = ("kurtosis", "skew")
"""They don't show in [our version of the stubs], but are possible in [`pyarrow>=20`].

[our version of the stubs]: https://github.com/narwhals-dev/narwhals/issues/2124#issuecomment-3191374210
[`pyarrow>=20`]: https://arrow.apache.org/docs/20.0/python/compute.html#grouped-aggregations
"""


class AggSpec:
    __slots__ = ("agg", "name", "option", "target")

    def __init__(
        self,
        target: acero.Target,
        agg: acero.Aggregation,
        option: acero.Opts = None,
        name: acero.OutputName = "",
    ) -> None:
        self.target = target
        self.agg = agg
        self.option = option
        self.name = name or str(target)

    @property
    def use_threads(self) -> bool:
        """See https://github.com/apache/arrow/issues/36709."""
        return acero.can_thread(self.agg)

    def __iter__(self) -> Iterator[acero.Target | acero.Aggregation | acero.Opts]:
        """Let's us duck-type as a 4-tuple."""
        yield from (self.target, self.agg, self.option, self.name)

    @classmethod
    def from_named_ir(cls, named_ir: NamedIR) -> Self:
        return cls.from_expr_ir(named_ir.expr, named_ir.name)

    @classmethod
    def from_agg_expr(cls, expr: agg.AggExpr, name: acero.OutputName) -> Self:
        tp = type(expr)
        if not (agg_name := SUPPORTED_AGG.get(tp)):
            raise group_by_error(name, expr)
        if not isinstance(expr.expr, ir.Column):
            raise group_by_error(name, expr, "too complex")
        option = (
            options.variance(expr.ddof)
            if isinstance(expr, (agg.Std, agg.Var))
            else options.AGG.get(tp)
        )
        return cls(expr.expr.name, agg_name, option, name)

    @classmethod
    def from_function_expr(cls, expr: ir.FunctionExpr, name: acero.OutputName) -> Self:
        tp = type(expr.function)
        if not (fn_name := SUPPORTED_FUNCTION.get(tp)):
            raise group_by_error(name, expr)
        args = expr.input
        if not (len(args) == 1 and isinstance(args[0], ir.Column)):
            raise group_by_error(name, expr, "too complex")
        return cls(args[0].name, fn_name, options.FUNCTION.get(tp), name)

    @classmethod
    def from_expr_ir(cls, expr: ir.ExprIR, name: acero.OutputName) -> Self:
        if is_agg_expr(expr):
            return cls.from_agg_expr(expr, name)
        if is_function_expr(expr):
            return cls.from_function_expr(expr, name)
        if not isinstance(expr, (ir.Len, ir.Column)):
            raise group_by_error(name, expr)
        fn_name = SUPPORTED_IR[type(expr)]
        return cls(expr.name if isinstance(expr, ir.Column) else (), fn_name, name=name)


def group_by_error(
    column_name: str, expr: ir.ExprIR, reason: Literal["too complex"] | None = None
) -> InvalidOperationError:
    backend = Implementation.PYARROW
    if reason == "too complex":
        msg = "Non-trivial complex aggregation found, which"
    else:
        msg = f"`{get_dispatch_name(expr)}()`"
    msg = f"{msg} is not supported in a `group_by` context for {backend!r}:\n{column_name}={expr!r}"
    return InvalidOperationError(msg)


def concat_str(native: pa.Table, *, separator: str = "") -> ChunkedArray:
    dtype = fn.string_type(native.schema.types)
    it = fn.cast_table(native, dtype).itercolumns()
    concat: Incomplete = pc.binary_join_element_wise
    join = options.join_replace_nulls()
    return concat(*it, fn.lit(separator, dtype), options=join)  # type: ignore[no-any-return]


class ArrowGroupBy(EagerDataFrameGroupBy["Frame"]):
    _df: Frame
    _keys: Seq[NamedIR]
    _key_names: Seq[str]
    _key_names_original: Seq[str]

    @property
    def compliant(self) -> Frame:
        return self._df

    def __iter__(self) -> Iterator[tuple[Any, Frame]]:
        from narwhals._plan.arrow.dataframe import partition_by

        by = self.key_names
        from_native = self.compliant._with_native
        for partition in partition_by(self.compliant.native, by):
            t = from_native(partition)
            yield (
                t.select_names(*by).row(0),
                t.select_names(*self._column_names_original),
            )

    def agg(self, irs: Seq[NamedIR]) -> Frame:
        compliant = self.compliant
        native = compliant.native
        key_names = self.key_names
        specs = (AggSpec.from_named_ir(e) for e in irs)
        result = compliant._with_native(acero.group_by_table(native, key_names, specs))
        if original := self._key_names_original:
            return result.rename(dict(zip(key_names, original)))
        return result
