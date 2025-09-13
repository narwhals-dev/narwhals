from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, overload

from narwhals._plan import common, expressions as ir
from narwhals._plan._guards import is_column, is_expr, is_series
from narwhals._plan._parse import (
    parse_into_expr_ir,
    parse_into_seq_of_expr_ir,
    parse_predicates_constraints_into_expr_ir,
)
from narwhals._plan.expressions import (
    aggregation as agg,
    boolean,
    functions as F,
    operators as ops,
)
from narwhals._plan.expressions.selectors import by_name
from narwhals._plan.expressions.window import Over
from narwhals._plan.options import (
    EWMOptions,
    RankOptions,
    SortMultipleOptions,
    SortOptions,
    rolling_options,
)
from narwhals._utils import Version
from narwhals.exceptions import ComputeError, InvalidOperationError

if TYPE_CHECKING:
    from typing_extensions import Never, Self

    from narwhals._plan.common import Function
    from narwhals._plan.expressions.categorical import ExprCatNamespace
    from narwhals._plan.expressions.lists import ExprListNamespace
    from narwhals._plan.expressions.name import ExprNameNamespace
    from narwhals._plan.expressions.strings import ExprStringNamespace
    from narwhals._plan.expressions.struct import ExprStructNamespace
    from narwhals._plan.expressions.temporal import ExprDateTimeNamespace
    from narwhals._plan.meta import IRMetaNamespace
    from narwhals._plan.typing import IntoExpr, IntoExprColumn, OneOrIterable, Seq, Udf
    from narwhals.typing import (
        ClosedInterval,
        FillNullStrategy,
        IntoDType,
        NumericLiteral,
        RankMethod,
        RollingInterpolationMethod,
        TemporalLiteral,
    )


# NOTE: Trying to keep consistent logic between `DataFrame.sort` and `Expr.sort_by`
def _parse_sort_by(
    by: OneOrIterable[IntoExpr] = (),
    *more_by: IntoExpr,
    descending: OneOrIterable[bool] = False,
    nulls_last: OneOrIterable[bool] = False,
) -> tuple[Seq[ir.ExprIR], SortMultipleOptions]:
    sort_by = parse_into_seq_of_expr_ir(by, *more_by)
    if length_changing := next((e for e in sort_by if e.is_scalar), None):
        msg = f"All expressions sort keys must preserve length, but got:\n{length_changing!r}"
        raise InvalidOperationError(msg)
    options = SortMultipleOptions.parse(descending=descending, nulls_last=nulls_last)
    return sort_by, options


# NOTE: Overly simplified placeholders for mocking typing
# Entirely ignoring namespace + function binding
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

    def exclude(self, *names: OneOrIterable[str]) -> Self:
        return self._from_ir(ir.Exclude.from_names(self._ir, *names))

    def count(self) -> Self:
        return self._from_ir(agg.Count(expr=self._ir))

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
        *partition_by: OneOrIterable[IntoExpr],
        order_by: OneOrIterable[IntoExpr] = None,
        descending: bool = False,
        nulls_last: bool = False,
    ) -> Self:
        node: ir.WindowExpr | ir.OrderedWindowExpr
        partition: Seq[ir.ExprIR] = ()
        if not (partition_by) and order_by is None:
            msg = "At least one of `partition_by` or `order_by` must be specified."
            raise TypeError(msg)
        if partition_by:
            partition = parse_into_seq_of_expr_ir(*partition_by)
        if order_by is not None:
            by = parse_into_seq_of_expr_ir(order_by)
            options = SortOptions(descending=descending, nulls_last=nulls_last)
            node = Over().to_ordered_window_expr(self._ir, partition, by, options)
        else:
            node = Over().to_window_expr(self._ir, partition)
        return self._from_ir(node)

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Self:
        options = SortOptions(descending=descending, nulls_last=nulls_last)
        return self._from_ir(ir.Sort(expr=self._ir, options=options))

    def sort_by(
        self,
        by: OneOrIterable[IntoExpr],
        *more_by: IntoExpr,
        descending: OneOrIterable[bool] = False,
        nulls_last: OneOrIterable[bool] = False,
    ) -> Self:
        keys, opts = _parse_sort_by(
            by, *more_by, descending=descending, nulls_last=nulls_last
        )
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
        include_breakpoint: bool = True,
    ) -> Self:
        node: F.Hist
        if bins is not None:
            if bin_count is not None:
                msg = "can only provide one of `bin_count` or `bins`"
                raise ComputeError(msg)
            node = F.HistBins(bins=tuple(bins), include_breakpoint=include_breakpoint)
        elif bin_count is not None:
            node = F.HistBinCount(
                bin_count=bin_count, include_breakpoint=include_breakpoint
            )
        else:
            node = F.HistBinCount(include_breakpoint=include_breakpoint)
        return self._with_unary(node)

    def log(self, base: float = math.e) -> Self:
        return self._with_unary(F.Log(base=base))

    def exp(self) -> Self:
        return self._with_unary(F.Exp())

    def sqrt(self) -> Self:
        return self._with_unary(F.Sqrt())

    def kurtosis(self, *, fisher: bool = True, bias: bool = True) -> Self:
        return self._with_unary(F.Kurtosis(fisher=fisher, bias=bias))

    def null_count(self) -> Self:
        return self._with_unary(F.NullCount())

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

    def mode(self) -> Self:
        return self._with_unary(F.Mode())

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
        it = parse_into_seq_of_expr_ir(lower_bound, upper_bound)
        return self._from_ir(F.Clip().to_function_expr(self._ir, *it))

    def cum_count(self, *, reverse: bool = False) -> Self:
        return self._with_unary(F.CumCount(reverse=reverse))

    def cum_min(self, *, reverse: bool = False) -> Self:
        return self._with_unary(F.CumMin(reverse=reverse))

    def cum_max(self, *, reverse: bool = False) -> Self:
        return self._with_unary(F.CumMax(reverse=reverse))

    def cum_prod(self, *, reverse: bool = False) -> Self:
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
        new: Sequence[Any] | None = None,
        *,
        return_dtype: IntoDType | None = None,
    ) -> Self:
        before: Seq[Any]
        after: Seq[Any]
        if new is None:
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
        function = F.ReplaceStrict(old=before, new=after, return_dtype=return_dtype)
        return self._with_unary(function)

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

    def any(self) -> Self:
        return self._with_unary(boolean.Any())

    def all(self) -> Self:
        return self._with_unary(boolean.All())

    def is_duplicated(self) -> Self:
        return self._with_unary(boolean.IsDuplicated())

    def is_finite(self) -> Self:
        return self._with_unary(boolean.IsFinite())

    def is_nan(self) -> Self:
        return self._with_unary(boolean.IsNan())

    def is_null(self) -> Self:
        return self._with_unary(boolean.IsNull())

    def is_first_distinct(self) -> Self:
        return self._with_unary(boolean.IsFirstDistinct())

    def is_last_distinct(self) -> Self:
        return self._with_unary(boolean.IsLastDistinct())

    def is_unique(self) -> Self:
        return self._with_unary(boolean.IsUnique())

    def is_between(
        self,
        lower_bound: IntoExpr,
        upper_bound: IntoExpr,
        closed: ClosedInterval = "both",
    ) -> Self:
        it = parse_into_seq_of_expr_ir(lower_bound, upper_bound)
        return self._from_ir(
            boolean.IsBetween(closed=closed).to_function_expr(self._ir, *it)
        )

    def is_in(self, other: Iterable[Any]) -> Self:
        if is_series(other):
            return self._with_unary(boolean.IsInSeries.from_series(other))
        if isinstance(other, Iterable):
            return self._with_unary(boolean.IsInSeq.from_iterable(other))
        if is_expr(other):
            return self._with_unary(boolean.IsInExpr(other=other._ir))
        msg = f"`is_in` only supports iterables, got: {type(other).__name__}"
        raise TypeError(msg)

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
        return self._with_unary(boolean.Not())

    @property
    def meta(self) -> IRMetaNamespace:
        from narwhals._plan.meta import IRMetaNamespace

        return IRMetaNamespace.from_expr(self)

    @property
    def name(self) -> ExprNameNamespace:
        """Specialized expressions for modifying the name of existing expressions.

        Examples:
            >>> from narwhals._plan import functions as nwd
            >>>
            >>> renamed = nwd.col("a", "b").name.suffix("_changed")
            >>> str(renamed._ir)
            "RenameAlias(expr=Columns(names=[a, b]), function=Suffix(suffix='_changed'))"
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


class Selector(Expr):
    _ir: ir.SelectorIR

    def __repr__(self) -> str:
        return f"nw._plan.Selector({self.version.name.lower()}):\n{self._ir!r}"

    @classmethod
    def _from_ir(cls, selector_ir: ir.SelectorIR, /) -> Self:  # type: ignore[override]
        obj = cls.__new__(cls)
        obj._ir = selector_ir
        return obj

    def _to_expr(self) -> Expr:
        return self._ir.to_narwhals(self.version)

    @overload  # type: ignore[override]
    def __or__(self, other: Self) -> Self: ...
    @overload
    def __or__(self, other: IntoExprColumn | int | bool) -> Expr: ...
    def __or__(self, other: IntoExprColumn | int | bool) -> Self | Expr:
        if isinstance(other, type(self)):
            op = ops.Or()
            return self._from_ir(op.to_binary_selector(self._ir, other._ir))
        return self._to_expr() | other

    @overload  # type: ignore[override]
    def __and__(self, other: Self) -> Self: ...
    @overload
    def __and__(self, other: IntoExprColumn | int | bool) -> Expr: ...
    def __and__(self, other: IntoExprColumn | int | bool) -> Self | Expr:
        if is_column(other) and (name := other.meta.output_name()):
            other = by_name(name)
        if isinstance(other, type(self)):
            op = ops.And()
            return self._from_ir(op.to_binary_selector(self._ir, other._ir))
        return self._to_expr() & other

    @overload  # type: ignore[override]
    def __sub__(self, other: Self) -> Self: ...
    @overload
    def __sub__(self, other: IntoExpr) -> Expr: ...
    def __sub__(self, other: IntoExpr) -> Self | Expr:
        if isinstance(other, type(self)):
            op = ops.Sub()
            return self._from_ir(op.to_binary_selector(self._ir, other._ir))
        return self._to_expr() - other

    @overload  # type: ignore[override]
    def __xor__(self, other: Self) -> Self: ...
    @overload
    def __xor__(self, other: IntoExprColumn | int | bool) -> Expr: ...
    def __xor__(self, other: IntoExprColumn | int | bool) -> Self | Expr:
        if isinstance(other, type(self)):
            op = ops.ExclusiveOr()
            return self._from_ir(op.to_binary_selector(self._ir, other._ir))
        return self._to_expr() ^ other

    def __invert__(self) -> Self:
        return self._from_ir(ir.InvertSelector(selector=self._ir))

    def __add__(self, other: Any) -> Expr:  # type: ignore[override]
        if isinstance(other, type(self)):
            msg = "unsupported operand type(s) for op: ('Selector' + 'Selector')"
            raise TypeError(msg)
        return self._to_expr() + other  # type: ignore[no-any-return]

    def __radd__(self, other: Any) -> Never:
        msg = "unsupported operand type(s) for op: ('Expr' + 'Selector')"
        raise TypeError(msg)

    def __rsub__(self, other: Any) -> Never:
        msg = "unsupported operand type(s) for op: ('Expr' - 'Selector')"
        raise TypeError(msg)

    @overload  # type: ignore[override]
    def __rand__(self, other: Self) -> Self: ...
    @overload
    def __rand__(self, other: IntoExprColumn | int | bool) -> Expr: ...
    def __rand__(self, other: IntoExprColumn | int | bool) -> Self | Expr:
        if is_column(other) and (name := other.meta.output_name()):
            return by_name(name) & self
        return self._to_expr().__rand__(other)

    @overload  # type: ignore[override]
    def __ror__(self, other: Self) -> Self: ...
    @overload
    def __ror__(self, other: IntoExprColumn | int | bool) -> Expr: ...
    def __ror__(self, other: IntoExprColumn | int | bool) -> Self | Expr:
        if is_column(other) and (name := other.meta.output_name()):
            return by_name(name) | self
        return self._to_expr().__ror__(other)

    @overload  # type: ignore[override]
    def __rxor__(self, other: Self) -> Self: ...
    @overload
    def __rxor__(self, other: IntoExprColumn | int | bool) -> Expr: ...
    def __rxor__(self, other: IntoExprColumn | int | bool) -> Self | Expr:
        if is_column(other) and (name := other.meta.output_name()):
            return by_name(name) ^ self
        return self._to_expr().__rxor__(other)


class ExprV1(Expr):
    _version: ClassVar[Version] = Version.V1


class SelectorV1(Selector):
    _version: ClassVar[Version] = Version.V1
