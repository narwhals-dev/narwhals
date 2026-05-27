from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import polars as pl

from narwhals._plan.common import todo
from narwhals._plan.compliant import CompliantExpr, typing as ct
from narwhals._plan.compliant.accessors import ExprStructNamespace
from narwhals._plan.polars import compat, functions as fn
from narwhals._plan.polars.classes import PolarsClasses
from narwhals._plan.polars.namespace import dtype_to_native, dtype_to_native_fast
from narwhals._utils import Version

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import TypeAlias

    from typing_extensions import Self

    from narwhals._plan import expressions as ir
    from narwhals._plan.expressions import (
        FunctionExpr as FExpr,
        HorizontalExpr as HExpr,
        aggregation as agg,
        boolean,
        functions as F,
    )
    from narwhals._plan.expressions.ranges import (
        DateRange,
        IntRange,
        LinearSpace,
        RangeFunction,
    )
    from narwhals._plan.expressions.strings import ConcatStr
    from narwhals._plan.expressions.struct import FieldByName
    from narwhals._plan.polars.dataframe import PolarsDataFrame as DataFrame  # noqa: F401
    from narwhals._plan.typing import NonNestedLiteralT_co, Seq2
    from narwhals.typing import IntoDType, PythonLiteral

__all__ = ("PolarsExpr",)

Incomplete: TypeAlias = Any

PolarsFrame: TypeAlias = "ct.Frame[pl.DataFrame, pl.Series, pl.LazyFrame]"


ExprT_co = TypeVar("ExprT_co", bound="PolarsExpr", covariant=True)


# TODO @dangotbanned: Do the long way first,
# then circle back on what pattern reduces the code the most
# (e.g maybe `pl.Expr.<method>` wrapping)
class PolarsExpr(CompliantExpr["DataFrame", pl.Expr, pl.Expr]):
    __slots__ = ("_native",)
    _native: pl.Expr
    version: ClassVar = Version.MAIN

    # NOTE: Unsure how much of `name` might be needed for polars
    @classmethod
    def from_native(cls, native: pl.Expr, name: str = "", /) -> Self:
        """`name` is only required for the inner-most `PolarsExpr` [^1].

        [^1]: Excluding any bugs in older versions.
        """
        obj = cls.__new__(cls)
        obj._native = native if not name else native.alias(name)
        return obj

    @classmethod
    def from_python(
        cls,
        value: PythonLiteral,
        name: str = "literal",
        /,
        *,
        dtype: IntoDType | None = None,
    ) -> Self:
        unknown = cls.version.dtypes.Unknown
        dtype_pl = None if dtype == unknown else dtype_to_native(dtype, cls.version)
        return cls.from_native(fn.lit(value, dtype_pl), name)

    @property
    def native(self) -> pl.Expr:
        return self._native

    @property
    def __narwhals_classes__(self) -> PolarsClasses:
        return PolarsClasses()

    def dispatch(self, node: ir.ExprIR, frame: PolarsFrame, name: str) -> PolarsExpr:
        """Trying to limit the API surface for now.

        - polars only uses `PolarsDataFrame._evaluate_irs`
        - pyarrow is more tangled up
        """
        return node.__expr_ir_dispatch__(node, self, frame, name)

    @classmethod
    def col(cls, node: ir.Column, _: Incomplete, name: str, /) -> Self:
        return cls.from_native(pl.col(node.name), name)

    @classmethod
    def lit(cls, node: ir.Lit[PythonLiteral], _: Incomplete, name: str, /) -> Self:
        return cls.from_python(node.value, name, dtype=node.dtype)

    @classmethod
    def lit_series(
        cls, node: ir.LitSeries[pl.Series], _: Incomplete, name: str, /
    ) -> Self:
        return cls.from_native(fn.lit(node.native), name)

    @classmethod
    def len_star(cls, _: ir.Len, __: Incomplete, name: str, /) -> Self:
        return cls.from_native(fn.len(), name)

    abs = todo()

    def all(self, node: FExpr[boolean.All], frame: Any, name: str) -> Self:
        return self.from_native(node.dispatch_arg(self, frame, name).native.all())

    def _dispatch_variadic_native(
        self, node: HExpr, frame: Any, name: str, /
    ) -> Iterator[pl.Expr]:
        exprs = iter(node.args)
        yield self.dispatch(next(exprs), frame, name).native
        for expr_ir in exprs:
            yield self.dispatch(expr_ir, frame, "").native

    def horizontal(
        self,
        node: HExpr,
        frame: Any,
        name: str,
        /,
        *,
        fill: Any = None,
        fn_native: Callable[..., pl.Expr] | None = None,
    ) -> Self:
        inputs = self._dispatch_variadic_native(node, frame, name)
        f = node.function
        kwds = f.to_dict()
        if fill is not None and kwds.pop("ignore_nulls", False):
            inputs = (e.fill_null(fill) for e in inputs)
        func = fn_native or getattr(pl, f.__expr_ir_dispatch__.name)
        return self.from_native(func(*inputs, **kwds), name)

    def all_horizontal(
        self, node: HExpr[boolean.AllHorizontal], frame: Any, name: str, /
    ) -> Self:
        return self.horizontal(node, frame, name, fill=True)

    def any_horizontal(
        self, node: HExpr[boolean.AnyHorizontal], frame: Any, name: str, /
    ) -> Self:
        return self.horizontal(node, frame, name, fill=False)

    coalesce = horizontal
    max_horizontal = horizontal
    min_horizontal = horizontal
    sum_horizontal = horizontal

    def concat_str(self, node: HExpr[ConcatStr], frame: Any, name: str) -> Self:
        return self.horizontal(node, frame, name, fn_native=fn.concat_str)

    def mean_horizontal(
        self, node: HExpr[F.MeanHorizontal], frame: Any, name: str, /
    ) -> Self:
        return self.horizontal(node, frame, name, fn_native=fn.mean_horizontal)

    def any(self, node: FExpr[boolean.Any], frame: Any, name: str) -> Self:
        return self.from_native(node.dispatch_arg(self, frame, name).native.any())

    def arg_max(self, node: agg.ArgMax, frame: Incomplete, name: str) -> Self:
        return self.from_native(self.dispatch(node.expr, frame, name).native.arg_max())

    def arg_min(self, node: agg.ArgMin, frame: Incomplete, name: str) -> Self:
        return self.from_native(self.dispatch(node.expr, frame, name).native.arg_min())

    def binary_expr(self, node: ir.BinaryExpr, frame: Any, name: str, /) -> Self:
        lhs, rhs = (
            self.dispatch(node.left, frame, name).native,
            self.dispatch(node.right, frame, "").native,
        )
        result: pl.Expr = node.op(lhs, rhs)
        return self.from_native(result, name)

    def cast(self, node: ir.Cast, frame: Incomplete, name: str) -> Self:
        dtype = dtype_to_native(node.dtype, self.version)
        return self.from_native(self.dispatch(node.expr, frame, name).native.cast(dtype))

    ceil = todo()
    clip = todo()
    clip_lower = todo()
    clip_upper = todo()

    def count(self, node: agg.Count, frame: Any, name: str) -> Self:
        return self.from_native(self.dispatch(node.expr, frame, name).native.count())

    cum_count = todo()
    cum_max = todo()
    cum_min = todo()
    cum_prod = todo()
    cum_sum = todo()
    diff = todo()

    def drop_nulls(self, node: FExpr[F.DropNulls], frame: Any, name: str) -> Self:
        return self.from_native(node.dispatch_arg(self, frame, name).native.drop_nulls())

    ewm_mean = todo()
    exp = todo()
    fill_nan = todo()

    def fill_null(self, node: FExpr[F.FillNull], frame: Any, name: str) -> Self:
        expr, value = node.dispatch_args(self, frame, name)
        return self.from_native(expr.native.fill_null(value.native))

    def fill_null_with_strategy(
        self, node: FExpr[F.FillNullWithStrategy], frame: Any, name: str
    ) -> Self:
        f = node.function
        native = node.dispatch_arg(self, frame, name).native
        return self.from_native(native.fill_null(strategy=f.strategy, limit=f.limit))

    def filter(self, node: ir.Filter, frame: Any, name: str, /) -> Self:
        native = self.dispatch(node.expr, frame, name).native
        result = native.filter(self.dispatch(node.by, frame, "").native)
        return self.from_native(result)

    def first(self, node: agg.First, frame: Any, name: str) -> Self:
        return self.from_native(self.dispatch(node.expr, frame, name).native.first())

    floor = todo()
    gather_every = todo()
    hist_bin_count = todo()
    hist_bins = todo()

    def _range_args(
        self, node: FExpr[RangeFunction[NonNestedLiteralT_co]], frame: Any
    ) -> Seq2[NonNestedLiteralT_co | pl.Expr]:
        if fastpath := node.function.try_unwrap_literals(node):
            return fastpath
        start, end = (e.native for e in node.dispatch_args(self, frame, ""))
        return start, end

    def date_range(self, node: FExpr[DateRange], frame: Any, name: str) -> Self:
        f = node.function
        interval = f"{f.interval}d"
        native = pl.date_range(*self._range_args(node, frame), interval, closed=f.closed)
        return self.from_native(native, name)

    def int_range(self, node: FExpr[IntRange], frame: Any, name: str) -> Self:
        f = node.function
        dtype = dtype_to_native_fast(f.dtype)
        native = pl.int_range(*self._range_args(node, frame), f.step, dtype=dtype)
        return self.from_native(native, name)

    def linear_space(self, node: FExpr[LinearSpace], frame: Any, name: str) -> Self:
        f = node.function
        n = f.num_samples
        native = fn.linear_space(*self._range_args(node, frame), n, closed=f.closed)
        return self.from_native(native, name)

    def is_between(
        self, node: FExpr[boolean.IsBetween], frame: Any, name: str, /
    ) -> Self:
        expr, lb, ub = node.dispatch_args(self, frame, name)
        result = expr.native.is_between(lb.native, ub.native, node.function.closed)
        return self.from_native(result, name)

    is_duplicated = todo()
    is_finite = todo()
    is_first_distinct = todo()

    def is_in_expr(self, node: FExpr[boolean.IsInExpr], frame: Any, name: str, /) -> Self:
        expr, other = node.dispatch_args(self, frame, name)
        result = expr.native.is_in(other.native)
        return self.from_native(result)

    def is_in_seq(self, node: FExpr[boolean.IsInSeq], frame: Any, name: str, /) -> Self:
        result = node.dispatch_arg(self, frame, name).native.is_in(node.function.other)
        return self.from_native(result)

    def is_in_series(
        self, node: FExpr[boolean.IsInSeries[pl.Series]], frame: Any, name: str, /
    ) -> Self:
        result = node.dispatch_arg(self, frame, name).native.is_in(
            node.function.other.native
        )
        return self.from_native(result)

    is_last_distinct = todo()
    is_nan = todo()
    is_not_nan = todo()

    def is_not_null(self, node: FExpr[boolean.IsNotNull], frame: Any, name: str) -> Self:
        return self.from_native(node.dispatch_arg(self, frame, name).native.is_not_null())

    def is_null(self, node: FExpr[boolean.IsNull], frame: Any, name: str) -> Self:
        return self.from_native(node.dispatch_arg(self, frame, name).native.is_null())

    is_unique = todo()
    kurtosis = todo()

    def last(self, node: agg.Last, frame: Any, name: str) -> Self:
        return self.from_native(self.dispatch(node.expr, frame, name).native.last())

    def len(self, node: agg.Len, frame: Any, name: str) -> Self:
        return self.from_native(self.dispatch(node.expr, frame, name).native.len())

    log = todo()

    def max(self, node: agg.Max, frame: Any, name: str) -> Self:
        return self.from_native(self.dispatch(node.expr, frame, name).native.max())

    def mean(self, node: agg.Mean, frame: Any, name: str) -> Self:
        return self.from_native(self.dispatch(node.expr, frame, name).native.mean())

    def median(self, node: agg.Median, frame: Any, name: str) -> Self:
        return self.from_native(self.dispatch(node.expr, frame, name).native.median())

    def min(self, node: agg.Min, frame: Any, name: str) -> Self:
        return self.from_native(self.dispatch(node.expr, frame, name).native.min())

    mode_all = todo()
    mode_any = todo()

    def n_unique(self, node: agg.NUnique, frame: Any, name: str) -> Self:
        return self.from_native(self.dispatch(node.expr, frame, name).native.n_unique())

    def not_(self, node: FExpr[boolean.Not], frame: Any, name: str) -> Self:
        return self.from_native(node.dispatch_arg(self, frame, name).native.not_())

    def null_count(self, node: FExpr[F.NullCount], frame: Any, name: str) -> Self:
        return self.from_native(node.dispatch_arg(self, frame, name).native.null_count())

    def over(self, node: ir.Over, frame: Any, name: str) -> Self:
        by = (self.dispatch(e, frame, "").native for e in node.partition_by)
        result = self.dispatch(node.expr, frame, name).native.over(*by)
        return self.from_native(result, name)

    def over_ordered(self, node: ir.OverOrdered, frame: Any, name: str) -> Self:
        by = (self.dispatch(e, frame, "").native for e in node.partition_by)
        result = fn.over(
            self.dispatch(node.expr, frame, "").native,
            *by,
            order_by=tuple(node.order_by_names()),
            descending=node.descending,
            nulls_last=node.nulls_last,
        )
        return self.from_native(result, name)

    def pow(self, node: FExpr[F.Pow], frame: Any, name: str) -> Self:
        base, exponent = node.dispatch_args(self, frame, name)
        return self.from_native(base.native.pow(exponent.native), name)

    def quantile(self, node: agg.Quantile, frame: Any, name: str) -> Self:
        return self.from_native(
            self.dispatch(node.expr, frame, name).native.quantile(
                node.quantile, node.interpolation
            )
        )

    rank = todo()
    replace_strict = todo()
    replace_strict_default = todo()
    rolling_sum = todo()
    rolling_mean = todo()
    rolling_std = todo()
    rolling_var = todo()
    round = todo()
    sample_frac = todo()
    sample_n = todo()
    shift = todo()
    skew = todo()

    def sort(self, node: ir.Sort, frame: Any, name: str) -> Self:
        native = self.dispatch(node.expr, frame, name).native
        result = native.sort(descending=node.descending, nulls_last=node.nulls_last)
        return self.from_native(result)

    def sort_by(self, node: ir.SortBy, frame: Any, name: str) -> Self:
        by = (self.dispatch(e, frame, "").native for e in node.by)
        native = self.dispatch(node.expr, frame, name).native
        result = native.sort_by(*by, **compat.sort(node.options, len(node.by)))
        return self.from_native(result)

    sqrt = todo()

    def std(self, node: agg.Std, frame: Any, name: str) -> Self:
        return self.from_native(
            self.dispatch(node.expr, frame, name).native.std(node.ddof)
        )

    def sum(self, node: agg.Sum, frame: Any, name: str) -> Self:
        return self.from_native(self.dispatch(node.expr, frame, name).native.sum())

    def ternary_expr(self, node: ir.TernaryExpr, frame: Any, name: str, /) -> Self:
        result = (
            pl.when(self.dispatch(node.predicate, frame, name).native)
            .then(self.dispatch(node.truthy, frame, "").native)
            .otherwise(self.dispatch(node.falsy, frame, "").native)
        )
        return self.from_native(result, name)

    unique = todo()

    def var(self, node: agg.Var, frame: Any, name: str) -> Self:
        return self.from_native(
            self.dispatch(node.expr, frame, name).native.var(node.ddof)
        )

    cat = todo()  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]
    dt = todo()  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]
    list = todo()  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]
    str = todo()  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]

    @property
    def struct(self) -> PolarsStructNamespace[Self]:
        return PolarsStructNamespace(self)


class PolarsStructNamespace(ExprStructNamespace[PolarsFrame, ExprT_co]):
    __slots__ = ("_compliant",)

    def __init__(self, compliant: ExprT_co, /) -> None:
        self._compliant: ExprT_co = compliant

    @property
    def compliant(self) -> ExprT_co:
        return self._compliant

    def field(
        self, node: FExpr[FieldByName], frame: PolarsFrame, name: str, /
    ) -> ExprT_co:
        compliant = self.compliant
        previous = node.dispatch_arg(compliant, frame, name).native
        return compliant.from_native(previous.struct.field(node.function.name), name)


PolarsExpr()
