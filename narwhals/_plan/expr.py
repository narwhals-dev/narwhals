from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from narwhals._plan import common, expressions as ir
from narwhals._plan._guards import is_expr, is_series
from narwhals._plan._parse import (
    parse_into_expr_ir,
    parse_into_seq_of_expr_ir,
    parse_predicates_constraints_into_expr_ir,
    parse_sort_by_into_seq_of_expr_ir,
)
from narwhals._plan.expressions import (
    aggregation as agg,
    functions as F,
    operators as ops,
)
from narwhals._plan.options import (
    EWMOptions,
    RankOptions,
    SortMultipleOptions,
    SortOptions,
    rolling_options,
)
from narwhals._typing_compat import deprecated
from narwhals._utils import Version, no_default, not_implemented
from narwhals.exceptions import ComputeError

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeVar

    from typing_extensions import Concatenate, ParamSpec, Self

    from narwhals._plan._function import Function
    from narwhals._plan.expressions.categorical import ExprCatNamespace
    from narwhals._plan.expressions.lists import ExprListNamespace
    from narwhals._plan.expressions.name import ExprNameNamespace
    from narwhals._plan.expressions.strings import ExprStringNamespace
    from narwhals._plan.expressions.struct import ExprStructNamespace
    from narwhals._plan.expressions.temporal import ExprDateTimeNamespace
    from narwhals._plan.meta import MetaNamespace
    from narwhals._plan.typing import IntoExpr, IntoExprColumn, OneOrIterable, Seq, Udf
    from narwhals._typing import NoDefault
    from narwhals.typing import (
        ClosedInterval,
        FillNullStrategy,
        IntoDType,
        ModeKeepStrategy,
        NumericLiteral,
        RankMethod,
        RollingInterpolationMethod,
        TemporalLiteral,
    )

    P = ParamSpec("P")
    R = TypeVar("R")


class Expr:
    _ir: ir.ExprIR
    _version: ClassVar[Version] = Version.MAIN

    def __repr__(self) -> str:
        return f"nw._plan.Expr({self.version.name.lower()}):\n{self._ir!r}"

    def __str__(self) -> str:
        """Use `print(self)` for formatting."""
        return f"nw._plan.Expr({self.version.name.lower()}):\n{self._ir!s}"

    def _repr_html_(self) -> str:
        return self._ir._repr_html_()

    @classmethod
    def _from_ir(cls, expr_ir: ir.ExprIR, /) -> Self:
        obj = cls.__new__(cls)
        obj._ir = expr_ir
        return obj

    @property
    def version(self) -> Version:
        return self._version

    def alias(self, name: str) -> Self:
        return self._from_ir(self._ir.alias(name))

    def cast(self, dtype: IntoDType) -> Self:
        return self._from_ir(self._ir.cast(common.into_dtype(dtype)))

    def exclude(self, *names: OneOrIterable[str]) -> Expr:
        from narwhals._plan import selectors as cs

        return (self.meta.as_selector() - cs.by_name(*names)).as_expr()

    def count(self) -> Self:
        return self._from_ir(agg.Count(expr=self._ir))

    def len(self) -> Self:
        return self._from_ir(agg.Len(expr=self._ir))

    def max(self) -> Self:
        return self._from_ir(agg.Max(expr=self._ir))

    def mean(self) -> Self:
        return self._from_ir(agg.Mean(expr=self._ir))

    def min(self) -> Self:
        return self._from_ir(agg.Min(expr=self._ir))

    def median(self) -> Self:
        return self._from_ir(agg.Median(expr=self._ir))

    def n_unique(self) -> Self:
        return self._from_ir(agg.NUnique(expr=self._ir))

    def sum(self) -> Self:
        return self._from_ir(agg.Sum(expr=self._ir))

    def arg_min(self) -> Self:
        return self._from_ir(agg.ArgMin(expr=self._ir))

    def arg_max(self) -> Self:
        return self._from_ir(agg.ArgMax(expr=self._ir))

    def first(self) -> Self:
        return self._from_ir(agg.First(expr=self._ir))

    def last(self) -> Self:
        return self._from_ir(agg.Last(expr=self._ir))

    def var(self, *, ddof: int = 1) -> Self:
        return self._from_ir(agg.Var(expr=self._ir, ddof=ddof))

    def std(self, *, ddof: int = 1) -> Self:
        return self._from_ir(agg.Std(expr=self._ir, ddof=ddof))

    def quantile(
        self, quantile: float, interpolation: RollingInterpolationMethod
    ) -> Self:
        return self._from_ir(
            agg.Quantile(expr=self._ir, quantile=quantile, interpolation=interpolation)
        )

    def over(
        self,
        *partition_by: OneOrIterable[IntoExprColumn],
        order_by: OneOrIterable[IntoExprColumn] | None = None,
        descending: bool = False,
        nulls_last: bool = False,
    ) -> Self:
        if not (partition_by) and order_by is None:
            msg = "At least one of `partition_by` or `order_by` must be specified."
            raise TypeError(msg)
        parse = parse_into_seq_of_expr_ir
        fn = self._ir
        group = parse(*partition_by) if partition_by else ()
        if order_by is None:
            return self._from_ir(ir.over(fn, group))
        over = ir.over_ordered
        order = parse(order_by)
        desc, nulls = descending, nulls_last
        return self._from_ir(over(fn, group, order, descending=desc, nulls_last=nulls))

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Self:
        options = SortOptions(descending=descending, nulls_last=nulls_last)
        return self._from_ir(ir.Sort(expr=self._ir, options=options))

    def sort_by(
        self,
        by: OneOrIterable[IntoExprColumn],
        *more_by: IntoExprColumn,
        descending: OneOrIterable[bool] = False,
        nulls_last: OneOrIterable[bool] = False,
    ) -> Self:
        keys = parse_sort_by_into_seq_of_expr_ir(by, *more_by)
        opts = SortMultipleOptions.parse(descending=descending, nulls_last=nulls_last)
        return self._from_ir(ir.SortBy(expr=self._ir, by=keys, options=opts))

    def filter(
        self, *predicates: OneOrIterable[IntoExprColumn], **constraints: Any
    ) -> Self:
        by = parse_predicates_constraints_into_expr_ir(*predicates, **constraints)
        return self._from_ir(ir.Filter(expr=self._ir, by=by))

    def _with_unary(self, function: Function, /) -> Self:
        return self._from_ir(function.to_function_expr(self._ir))

    def abs(self) -> Self:
        return self._with_unary(F.Abs())

    def hist(
        self,
        bins: Sequence[float] | None = None,
        *,
        bin_count: int | None = None,
        include_breakpoint: bool = False,
        include_category: bool = False,
    ) -> Self:
        if include_category:
            msg = f"`Expr.hist({include_category=})` is not yet implemented"
            raise NotImplementedError(msg)
        node: F.Hist
        if bins is not None:
            if bin_count is not None:
                msg = "can only provide one of `bin_count` or `bins`"
                raise ComputeError(msg)
            node = F.Hist.from_bins(bins, include_breakpoint=include_breakpoint)
        elif bin_count is not None:
            node = F.Hist.from_bin_count(bin_count, include_breakpoint=include_breakpoint)
        else:
            node = F.Hist.from_bin_count(include_breakpoint=include_breakpoint)
        return self._with_unary(node)

    def log(self, base: float = math.e) -> Self:
        return self._with_unary(F.Log(base=base))

    def exp(self) -> Self:
        return self._with_unary(F.Exp())

    def sqrt(self) -> Self:
        return self._with_unary(F.Sqrt())

    def kurtosis(self) -> Self:
        return self._with_unary(F.Kurtosis())

    def null_count(self) -> Self:
        return self._with_unary(F.NullCount())

    def fill_nan(self, value: float | Self | None) -> Self:
        fill_value = parse_into_expr_ir(value, str_as_lit=True)
        root = self._ir
        if any(e.meta.has_multiple_outputs() for e in (root, fill_value)):
            return self._from_ir(F.FillNan().to_function_expr(root, fill_value))
        # https://github.com/pola-rs/polars/blob/e1d6f294218a36497255e2d872c223e19a47e2ec/crates/polars-plan/src/dsl/mod.rs#L894-L902
        predicate = self.is_not_nan() | self.is_null()
        return self._from_ir(ir.ternary_expr(predicate._ir, root, fill_value))

    def fill_null(
        self,
        value: IntoExpr = None,
        strategy: FillNullStrategy | None = None,
        limit: int | None = None,
    ) -> Self:
        if strategy is None:
            e = parse_into_expr_ir(value, str_as_lit=True)
            return self._from_ir(F.FillNull().to_function_expr(self._ir, e))
        return self._with_unary(F.FillNullWithStrategy(strategy=strategy, limit=limit))

    def shift(self, n: int) -> Self:
        return self._with_unary(F.Shift(n=n))

    def drop_nulls(self) -> Self:
        return self._with_unary(F.DropNulls())

    def mode(self, *, keep: ModeKeepStrategy = "all") -> Self:
        if func := {"all": F.ModeAll, "any": F.ModeAny}.get(keep):
            return self._with_unary(func())
        msg = f"`keep` must be one of ('all', 'any'), but got {keep!r}"
        raise TypeError(msg)

    def skew(self) -> Self:
        return self._with_unary(F.Skew())

    def rank(self, method: RankMethod = "average", *, descending: bool = False) -> Self:
        options = RankOptions(method=method, descending=descending)
        return self._with_unary(F.Rank(options=options))

    def clip(
        self,
        lower_bound: IntoExprColumn | NumericLiteral | TemporalLiteral | None = None,
        upper_bound: IntoExprColumn | NumericLiteral | TemporalLiteral | None = None,
    ) -> Self:
        f: ir.FunctionExpr
        if upper_bound is None:
            f = F.ClipLower().to_function_expr(self._ir, parse_into_expr_ir(lower_bound))
        elif lower_bound is None:
            f = F.ClipUpper().to_function_expr(self._ir, parse_into_expr_ir(upper_bound))
        else:
            it = parse_into_seq_of_expr_ir(lower_bound, upper_bound)
            f = F.Clip().to_function_expr(self._ir, *it)
        return self._from_ir(f)

    def cum_count(self, *, reverse: bool = False) -> Self:  # pragma: no cover
        return self._with_unary(F.CumCount(reverse=reverse))

    def cum_min(self, *, reverse: bool = False) -> Self:  # pragma: no cover
        return self._with_unary(F.CumMin(reverse=reverse))

    def cum_max(self, *, reverse: bool = False) -> Self:  # pragma: no cover
        return self._with_unary(F.CumMax(reverse=reverse))

    def cum_prod(self, *, reverse: bool = False) -> Self:  # pragma: no cover
        return self._with_unary(F.CumProd(reverse=reverse))

    def cum_sum(self, *, reverse: bool = False) -> Self:
        return self._with_unary(F.CumSum(reverse=reverse))

    def rolling_sum(
        self, window_size: int, *, min_samples: int | None = None, center: bool = False
    ) -> Self:
        options = rolling_options(window_size, min_samples, center=center)
        return self._with_unary(F.RollingSum(options=options))

    def rolling_mean(
        self, window_size: int, *, min_samples: int | None = None, center: bool = False
    ) -> Self:
        options = rolling_options(window_size, min_samples, center=center)
        return self._with_unary(F.RollingMean(options=options))

    def rolling_var(
        self,
        window_size: int,
        *,
        min_samples: int | None = None,
        center: bool = False,
        ddof: int = 1,
    ) -> Self:
        options = rolling_options(window_size, min_samples, center=center, ddof=ddof)
        return self._with_unary(F.RollingVar(options=options))

    def rolling_std(
        self,
        window_size: int,
        *,
        min_samples: int | None = None,
        center: bool = False,
        ddof: int = 1,
    ) -> Self:
        options = rolling_options(window_size, min_samples, center=center, ddof=ddof)
        return self._with_unary(F.RollingStd(options=options))

    def diff(self) -> Self:
        return self._with_unary(F.Diff())

    def unique(self) -> Self:
        return self._with_unary(F.Unique())

    def round(self, decimals: int = 0) -> Self:
        return self._with_unary(F.Round(decimals=decimals))

    def ceil(self) -> Self:
        return self._with_unary(F.Ceil())

    def floor(self) -> Self:
        return self._with_unary(F.Floor())

    def ewm_mean(
        self,
        *,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
        adjust: bool = True,
        min_samples: int = 1,
        ignore_nulls: bool = False,
    ) -> Self:
        options = EWMOptions(
            com=com,
            span=span,
            half_life=half_life,
            alpha=alpha,
            adjust=adjust,
            min_samples=min_samples,
            ignore_nulls=ignore_nulls,
        )
        return self._with_unary(F.EwmMean(options=options))

    def replace_strict(
        self,
        old: Sequence[Any] | Mapping[Any, Any],
        new: Sequence[Any] | NoDefault = no_default,
        *,
        default: IntoExpr | NoDefault = no_default,
        return_dtype: IntoDType | None = None,
    ) -> Self:
        before: Seq[Any]
        after: Seq[Any]
        if new is no_default:
            if not isinstance(old, Mapping):
                msg = "`new` argument is required if `old` argument is not a Mapping type"
                raise TypeError(msg)
            before = tuple(old)
            after = tuple(old.values())
        elif isinstance(old, Mapping):
            msg = "`new` argument cannot be used if `old` argument is a Mapping type"
            raise TypeError(msg)
        else:
            before = tuple(old)
            after = tuple(new)
        if return_dtype is not None:
            return_dtype = common.into_dtype(return_dtype)

        if default is no_default:
            function = F.ReplaceStrict(old=before, new=after, return_dtype=return_dtype)
            return self._with_unary(function)
        function = F.ReplaceStrictDefault(
            old=before, new=after, return_dtype=return_dtype
        )
        default_ir = parse_into_expr_ir(default, str_as_lit=True)
        return self._from_ir(function.to_function_expr(self._ir, default_ir))

    def gather_every(self, n: int, offset: int = 0) -> Self:
        return self._with_unary(F.GatherEvery(n=n, offset=offset))

    def map_batches(
        self,
        function: Udf,
        return_dtype: IntoDType | None = None,
        *,
        is_elementwise: bool = False,
        returns_scalar: bool = False,
    ) -> Self:
        if return_dtype is not None:
            return_dtype = common.into_dtype(return_dtype)
        return self._with_unary(
            F.MapBatches(
                function=function,
                return_dtype=return_dtype,
                is_elementwise=is_elementwise,
                returns_scalar=returns_scalar,
            )
        )

    # TODO @dangotbanned: Come back to this when *properly* building out `Version` support
    @deprecated("Use `v1.Expr.sample` or `{DataFrame,Series}.sample` instead")
    def sample(
        self,
        n: int | None = None,
        *,
        fraction: float | None = None,
        with_replacement: bool = False,
        seed: int | None = None,
    ) -> Self:
        f = F.sample(n, fraction=fraction, with_replacement=with_replacement, seed=seed)
        return self._with_unary(f)

    def any(self) -> Self:
        return self._with_unary(ir.boolean.Any())

    def all(self) -> Self:
        return self._with_unary(ir.boolean.All())

    def is_duplicated(self) -> Self:
        return self._with_unary(ir.boolean.IsDuplicated())

    def is_finite(self) -> Self:
        return self._with_unary(ir.boolean.IsFinite())

    def is_nan(self) -> Self:
        return self._with_unary(ir.boolean.IsNan())

    def is_null(self) -> Self:
        return self._with_unary(ir.boolean.IsNull())

    def is_not_nan(self) -> Self:
        return self._with_unary(ir.boolean.IsNotNan())

    def is_not_null(self) -> Self:
        return self._with_unary(ir.boolean.IsNotNull())

    def is_first_distinct(self) -> Self:
        return self._with_unary(ir.boolean.IsFirstDistinct())

    def is_last_distinct(self) -> Self:
        return self._with_unary(ir.boolean.IsLastDistinct())

    def is_unique(self) -> Self:
        return self._with_unary(ir.boolean.IsUnique())

    def is_between(
        self,
        lower_bound: IntoExpr,
        upper_bound: IntoExpr,
        closed: ClosedInterval = "both",
    ) -> Self:
        it = parse_into_seq_of_expr_ir(lower_bound, upper_bound)
        return self._from_ir(
            ir.boolean.IsBetween(closed=closed).to_function_expr(self._ir, *it)
        )

    def is_in(self, other: Iterable[Any] | Expr) -> Self:
        if is_series(other):
            return self._with_unary(ir.boolean.IsInSeries.from_series(other))
        if isinstance(other, Iterable):
            return self._with_unary(ir.boolean.IsInSeq.from_iterable(other))
        if is_expr(other):
            return self._from_ir(
                ir.boolean.IsInExpr().to_function_expr(self._ir, other._ir)
            )
        msg = f"`is_in` only supports iterables or Expr, got: {type(other).__name__}"
        raise TypeError(msg)

    def pipe(
        self, function: Callable[Concatenate[Self, P], R], *args: P.args, **kwds: P.kwargs
    ) -> R:
        return function(self, *args, **kwds)

    def _with_binary(
        self,
        op: type[ops.Operator],
        other: IntoExpr,
        *,
        str_as_lit: bool = False,
        reflect: bool = False,
    ) -> Self:
        other_ir = parse_into_expr_ir(other, str_as_lit=str_as_lit)
        args = (self._ir, other_ir) if not reflect else (other_ir, self._ir)
        return self._from_ir(op().to_binary_expr(*args))

    def __eq__(self, other: IntoExpr) -> Self:  # type: ignore[override]
        return self._with_binary(ops.Eq, other, str_as_lit=True)

    def __ne__(self, other: IntoExpr) -> Self:  # type: ignore[override]
        return self._with_binary(ops.NotEq, other, str_as_lit=True)

    def __lt__(self, other: IntoExpr) -> Self:
        return self._with_binary(ops.Lt, other, str_as_lit=True)

    def __le__(self, other: IntoExpr) -> Self:
        return self._with_binary(ops.LtEq, other, str_as_lit=True)

    def __gt__(self, other: IntoExpr) -> Self:
        return self._with_binary(ops.Gt, other, str_as_lit=True)

    def __ge__(self, other: IntoExpr) -> Self:
        return self._with_binary(ops.GtEq, other, str_as_lit=True)

    def __add__(self, other: IntoExpr) -> Self:
        return self._with_binary(ops.Add, other, str_as_lit=True)

    def __radd__(self, other: IntoExpr) -> Self:
        return self._with_binary(ops.Add, other, str_as_lit=True, reflect=True)

    def __sub__(self, other: IntoExpr) -> Self:
        return self._with_binary(ops.Sub, other)

    def __rsub__(self, other: IntoExpr) -> Self:
        return self._with_binary(ops.Sub, other, reflect=True)

    def __mul__(self, other: IntoExpr) -> Self:
        return self._with_binary(ops.Multiply, other)

    def __rmul__(self, other: IntoExpr) -> Self:
        return self._with_binary(ops.Multiply, other, reflect=True)

    def __truediv__(self, other: IntoExpr) -> Self:
        return self._with_binary(ops.TrueDivide, other)

    def __rtruediv__(self, other: IntoExpr) -> Self:
        return self._with_binary(ops.TrueDivide, other, reflect=True)

    def __floordiv__(self, other: IntoExpr) -> Self:
        return self._with_binary(ops.FloorDivide, other)

    def __rfloordiv__(self, other: IntoExpr) -> Self:
        return self._with_binary(ops.FloorDivide, other, reflect=True)

    def __mod__(self, other: IntoExpr) -> Self:
        return self._with_binary(ops.Modulus, other)

    def __rmod__(self, other: IntoExpr) -> Self:
        return self._with_binary(ops.Modulus, other, reflect=True)

    def __and__(self, other: IntoExprColumn | int | bool) -> Self:
        return self._with_binary(ops.And, other)

    def __rand__(self, other: IntoExprColumn | int | bool) -> Self:
        return self._with_binary(ops.And, other, reflect=True)

    def __or__(self, other: IntoExprColumn | int | bool) -> Self:
        return self._with_binary(ops.Or, other)

    def __ror__(self, other: IntoExprColumn | int | bool) -> Self:
        return self._with_binary(ops.Or, other, reflect=True)

    def __xor__(self, other: IntoExprColumn | int | bool) -> Self:
        return self._with_binary(ops.ExclusiveOr, other)

    def __rxor__(self, other: IntoExprColumn | int | bool) -> Self:
        return self._with_binary(ops.ExclusiveOr, other, reflect=True)

    def __pow__(self, exponent: IntoExprColumn | float) -> Self:
        exp = parse_into_expr_ir(exponent)
        return self._from_ir(F.Pow().to_function_expr(self._ir, exp))

    def __rpow__(self, base: IntoExprColumn | float) -> Self:
        return self._from_ir(F.Pow().to_function_expr(parse_into_expr_ir(base), self._ir))

    def __invert__(self) -> Self:
        return self._with_unary(ir.boolean.Not())

    @property
    def meta(self) -> MetaNamespace:
        from narwhals._plan.meta import MetaNamespace

        return MetaNamespace.from_expr(self)

    @property
    def name(self) -> ExprNameNamespace:
        """Specialized expressions for modifying the name of existing expressions.

        Examples:
            >>> from narwhals import _plan as nw
            >>>
            >>> renamed = nw.col("a", "b").name.suffix("_changed")
            >>> str(renamed._ir)
            "RenameAlias(expr=RootSelector(selector=ByName(names=[a, b], require_all=True)), function=Suffix(suffix='_changed'))"
        """
        from narwhals._plan.expressions.name import ExprNameNamespace

        return ExprNameNamespace(_expr=self)

    @property
    def cat(self) -> ExprCatNamespace:
        from narwhals._plan.expressions.categorical import ExprCatNamespace

        return ExprCatNamespace(_expr=self)

    @property
    def struct(self) -> ExprStructNamespace:
        from narwhals._plan.expressions.struct import ExprStructNamespace

        return ExprStructNamespace(_expr=self)

    @property
    def dt(self) -> ExprDateTimeNamespace:
        from narwhals._plan.expressions.temporal import ExprDateTimeNamespace

        return ExprDateTimeNamespace(_expr=self)

    @property
    def list(self) -> ExprListNamespace:
        from narwhals._plan.expressions.lists import ExprListNamespace

        return ExprListNamespace(_expr=self)

    @property
    def str(self) -> ExprStringNamespace:
        from narwhals._plan.expressions.strings import ExprStringNamespace

        return ExprStringNamespace(_expr=self)

    is_close = not_implemented()
    head = not_implemented()
    tail = not_implemented()


class ExprV1(Expr):
    _version: ClassVar[Version] = Version.V1
