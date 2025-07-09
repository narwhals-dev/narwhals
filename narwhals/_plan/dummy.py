"""Mock version of current narwhals API."""

from __future__ import annotations

import math
import typing as t
from typing import TYPE_CHECKING, Generic

from narwhals._plan import (
    aggregation as agg,
    boolean,
    expr,
    expr_expansion,
    expr_parsing as parse,
    functions as F,  # noqa: N812
    operators as ops,
)
from narwhals._plan.common import NamedIR, is_column, is_expr, is_series
from narwhals._plan.contexts import ExprContext
from narwhals._plan.options import (
    EWMOptions,
    RankOptions,
    RollingOptionsFixedWindow,
    RollingVarParams,
    SortMultipleOptions,
    SortOptions,
)
from narwhals._plan.selectors import by_name
from narwhals._plan.typing import NativeFrameT, NativeSeriesT
from narwhals._plan.window import Over
from narwhals._utils import Version
from narwhals.dependencies import is_pyarrow_chunked_array, is_pyarrow_table
from narwhals.dtypes import DType
from narwhals.exceptions import ComputeError, InvalidOperationError
from narwhals.schema import Schema

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import pyarrow as pa
    from typing_extensions import Never, Self, TypeAlias

    from narwhals._plan.categorical import ExprCatNamespace
    from narwhals._plan.common import ExprIR
    from narwhals._plan.lists import ExprListNamespace
    from narwhals._plan.meta import IRMetaNamespace
    from narwhals._plan.name import ExprNameNamespace
    from narwhals._plan.protocols import DummyCompliantFrame, DummyCompliantSeries
    from narwhals._plan.schema import FrozenSchema
    from narwhals._plan.strings import ExprStringNamespace
    from narwhals._plan.struct import ExprStructNamespace
    from narwhals._plan.temporal import ExprDateTimeNamespace
    from narwhals._plan.typing import ExprT, IntoExpr, IntoExprColumn, Ns, Seq, Udf
    from narwhals.typing import (
        ClosedInterval,
        FillNullStrategy,
        IntoDType,
        NativeFrame,
        NativeSeries,
        NumericLiteral,
        RankMethod,
        RollingInterpolationMethod,
        TemporalLiteral,
    )


CompliantFrame: TypeAlias = "DummyCompliantFrame[t.Any, NativeFrameT, NativeSeriesT]"


# NOTE: Trying to keep consistent logic between `DataFrame.sort` and `Expr.sort_by`
def _parse_sort_by(
    by: IntoExpr | Iterable[IntoExpr] = (),
    *more_by: IntoExpr,
    descending: bool | t.Iterable[bool] = False,
    nulls_last: bool | t.Iterable[bool] = False,
) -> tuple[Seq[ExprIR], SortMultipleOptions]:
    sort_by = parse.parse_into_seq_of_expr_ir(by, *more_by)
    if length_changing := next((e for e in sort_by if e.is_scalar), None):
        msg = f"All expressions sort keys must preserve length, but got:\n{length_changing!r}"
        raise InvalidOperationError(msg)
    options = SortMultipleOptions.parse(descending=descending, nulls_last=nulls_last)
    return sort_by, options


# NOTE: Overly simplified placeholders for mocking typing
# Entirely ignoring namespace + function binding
class DummyExpr:
    _ir: ExprIR
    _version: t.ClassVar[Version] = Version.MAIN

    def __repr__(self) -> str:
        return f"Narwhals DummyExpr ({self.version.name.lower()}):\n{self._ir!r}"

    def __str__(self) -> str:
        """Use `print(self)` for formatting."""
        return f"Narwhals DummyExpr ({self.version.name.lower()}):\n{self._ir!s}"

    def _repr_html_(self) -> str:
        return self._ir._repr_html_()

    @classmethod
    def _from_ir(cls, ir: ExprIR, /) -> Self:
        obj = cls.__new__(cls)
        obj._ir = ir
        return obj

    def _to_compliant(self, plx: Ns[ExprT], /) -> ExprT:
        return self._ir.to_compliant(plx)

    @property
    def version(self) -> Version:
        return self._version

    def alias(self, name: str) -> Self:
        return self._from_ir(expr.Alias(expr=self._ir, name=name))

    def cast(self, dtype: DType | type[DType]) -> Self:
        dtype = dtype if isinstance(dtype, DType) else self.version.dtypes.Unknown()
        return self._from_ir(self._ir.cast(dtype))

    def exclude(self, *names: str | t.Iterable[str]) -> Self:
        return self._from_ir(expr.Exclude.from_names(self._ir, *names))

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
        *partition_by: IntoExpr | t.Iterable[IntoExpr],
        order_by: IntoExpr | t.Iterable[IntoExpr] = None,
        descending: bool = False,
        nulls_last: bool = False,
    ) -> Self:
        node: expr.WindowExpr | expr.OrderedWindowExpr
        partition: Seq[ExprIR] = ()
        if not (partition_by) and order_by is None:
            msg = "At least one of `partition_by` or `order_by` must be specified."
            raise TypeError(msg)
        if partition_by:
            partition = parse.parse_into_seq_of_expr_ir(*partition_by)
        if order_by is not None:
            by = parse.parse_into_seq_of_expr_ir(order_by)
            options = SortOptions(descending=descending, nulls_last=nulls_last)
            node = Over().to_ordered_window_expr(self._ir, partition, by, options)
        else:
            node = Over().to_window_expr(self._ir, partition)
        return self._from_ir(node)

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Self:
        options = SortOptions(descending=descending, nulls_last=nulls_last)
        return self._from_ir(expr.Sort(expr=self._ir, options=options))

    def sort_by(
        self,
        by: IntoExpr | t.Iterable[IntoExpr],
        *more_by: IntoExpr,
        descending: bool | t.Iterable[bool] = False,
        nulls_last: bool | t.Iterable[bool] = False,
    ) -> Self:
        keys, opts = _parse_sort_by(
            by, *more_by, descending=descending, nulls_last=nulls_last
        )
        return self._from_ir(expr.SortBy(expr=self._ir, by=keys, options=opts))

    def filter(
        self,
        *predicates: IntoExprColumn | t.Iterable[IntoExprColumn],
        **constraints: t.Any,
    ) -> Self:
        by = parse.parse_predicates_constraints_into_expr_ir(*predicates, **constraints)
        return self._from_ir(expr.Filter(expr=self._ir, by=by))

    def abs(self) -> Self:
        return self._from_ir(F.Abs().to_function_expr(self._ir))

    def hist(
        self,
        bins: t.Sequence[float] | None = None,
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
        return self._from_ir(node.to_function_expr(self._ir))

    def log(self, base: float = math.e) -> Self:
        return self._from_ir(F.Log(base=base).to_function_expr(self._ir))

    def exp(self) -> Self:
        return self._from_ir(F.Exp().to_function_expr(self._ir))

    def sqrt(self) -> Self:
        return self._from_ir(F.Sqrt().to_function_expr(self._ir))

    def kurtosis(self, *, fisher: bool = True, bias: bool = True) -> Self:
        return self._from_ir(
            F.Kurtosis(fisher=fisher, bias=bias).to_function_expr(self._ir)
        )

    def null_count(self) -> Self:
        return self._from_ir(F.NullCount().to_function_expr(self._ir))

    def fill_null(
        self,
        value: IntoExpr = None,
        strategy: FillNullStrategy | None = None,
        limit: int | None = None,
    ) -> Self:
        if strategy is None:
            ir = parse.parse_into_expr_ir(value, str_as_lit=True)
            return self._from_ir(F.FillNull().to_function_expr(self._ir, ir))
        fill = F.FillNullWithStrategy(strategy=strategy, limit=limit)
        return self._from_ir(fill.to_function_expr(self._ir))

    def shift(self, n: int) -> Self:
        return self._from_ir(F.Shift(n=n).to_function_expr(self._ir))

    def drop_nulls(self) -> Self:
        return self._from_ir(F.DropNulls().to_function_expr(self._ir))

    def mode(self) -> Self:
        return self._from_ir(F.Mode().to_function_expr(self._ir))

    def skew(self) -> Self:
        return self._from_ir(F.Skew().to_function_expr(self._ir))

    def rank(self, method: RankMethod = "average", *, descending: bool = False) -> Self:
        options = RankOptions(method=method, descending=descending)
        return self._from_ir(F.Rank(options=options).to_function_expr(self._ir))

    def clip(
        self,
        lower_bound: IntoExprColumn | NumericLiteral | TemporalLiteral | None = None,
        upper_bound: IntoExprColumn | NumericLiteral | TemporalLiteral | None = None,
    ) -> Self:
        return self._from_ir(
            F.Clip().to_function_expr(
                self._ir, *parse.parse_into_seq_of_expr_ir(lower_bound, upper_bound)
            )
        )

    def cum_count(self, *, reverse: bool = False) -> Self:
        return self._from_ir(F.CumCount(reverse=reverse).to_function_expr(self._ir))

    def cum_min(self, *, reverse: bool = False) -> Self:
        return self._from_ir(F.CumMin(reverse=reverse).to_function_expr(self._ir))

    def cum_max(self, *, reverse: bool = False) -> Self:
        return self._from_ir(F.CumMax(reverse=reverse).to_function_expr(self._ir))

    def cum_prod(self, *, reverse: bool = False) -> Self:
        return self._from_ir(F.CumProd(reverse=reverse).to_function_expr(self._ir))

    def cum_sum(self, *, reverse: bool = False) -> Self:
        return self._from_ir(F.CumSum(reverse=reverse).to_function_expr(self._ir))

    def rolling_sum(
        self, window_size: int, *, min_samples: int | None = None, center: bool = False
    ) -> Self:
        min_samples = window_size if min_samples is None else min_samples
        fn_params = None
        options = RollingOptionsFixedWindow(
            window_size=window_size,
            min_samples=min_samples,
            center=center,
            fn_params=fn_params,
        )
        function = F.RollingSum(options=options)
        return self._from_ir(function.to_function_expr(self._ir))

    def rolling_mean(
        self, window_size: int, *, min_samples: int | None = None, center: bool = False
    ) -> Self:
        min_samples = window_size if min_samples is None else min_samples
        fn_params = None
        options = RollingOptionsFixedWindow(
            window_size=window_size,
            min_samples=min_samples,
            center=center,
            fn_params=fn_params,
        )
        function = F.RollingMean(options=options)
        return self._from_ir(function.to_function_expr(self._ir))

    def rolling_var(
        self,
        window_size: int,
        *,
        min_samples: int | None = None,
        center: bool = False,
        ddof: int = 1,
    ) -> Self:
        min_samples = window_size if min_samples is None else min_samples
        fn_params = RollingVarParams(ddof=ddof)
        options = RollingOptionsFixedWindow(
            window_size=window_size,
            min_samples=min_samples,
            center=center,
            fn_params=fn_params,
        )
        function = F.RollingVar(options=options)
        return self._from_ir(function.to_function_expr(self._ir))

    def rolling_std(
        self,
        window_size: int,
        *,
        min_samples: int | None = None,
        center: bool = False,
        ddof: int = 1,
    ) -> Self:
        min_samples = window_size if min_samples is None else min_samples
        fn_params = RollingVarParams(ddof=ddof)
        options = RollingOptionsFixedWindow(
            window_size=window_size,
            min_samples=min_samples,
            center=center,
            fn_params=fn_params,
        )
        function = F.RollingStd(options=options)
        return self._from_ir(function.to_function_expr(self._ir))

    def diff(self) -> Self:
        return self._from_ir(F.Diff().to_function_expr(self._ir))

    def unique(self) -> Self:
        return self._from_ir(F.Unique().to_function_expr(self._ir))

    def round(self, decimals: int = 0) -> Self:
        return self._from_ir(F.Round(decimals=decimals).to_function_expr(self._ir))

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
        return self._from_ir(F.EwmMean(options=options).to_function_expr(self._ir))

    def replace_strict(
        self,
        old: t.Sequence[t.Any] | t.Mapping[t.Any, t.Any],
        new: t.Sequence[t.Any] | None = None,
        *,
        return_dtype: IntoDType | None = None,
    ) -> Self:
        before: Seq[t.Any]
        after: Seq[t.Any]
        if new is None:
            if not isinstance(old, t.Mapping):
                msg = "`new` argument is required if `old` argument is not a Mapping type"
                raise TypeError(msg)
            before = tuple(old)
            after = tuple(old.values())
        elif isinstance(old, t.Mapping):
            # NOTE: polars raises later when this occurs
            # TypeError: cannot create expression literal for value of type dict.
            # Hint: Pass `allow_object=True` to accept any value and create a literal of type Object.
            msg = "`new` argument cannot be used if `old` argument is a Mapping type"
            raise TypeError(msg)
        else:
            before = tuple(old)
            after = tuple(new)
        function = F.ReplaceStrict(old=before, new=after, return_dtype=return_dtype)
        return self._from_ir(function.to_function_expr(self._ir))

    def gather_every(self, n: int, offset: int = 0) -> Self:
        return self._from_ir(F.GatherEvery(n=n, offset=offset).to_function_expr(self._ir))

    def map_batches(
        self,
        function: Udf,
        return_dtype: IntoDType | None = None,
        *,
        is_elementwise: bool = False,
        returns_scalar: bool = False,
    ) -> Self:
        return self._from_ir(
            F.MapBatches(
                function=function,
                return_dtype=return_dtype,
                is_elementwise=is_elementwise,
                returns_scalar=returns_scalar,
            ).to_function_expr(self._ir)
        )

    def any(self) -> Self:
        return self._from_ir(boolean.Any().to_function_expr(self._ir))

    def all(self) -> Self:
        return self._from_ir(boolean.All().to_function_expr(self._ir))

    def is_duplicated(self) -> Self:
        return self._from_ir(boolean.IsDuplicated().to_function_expr(self._ir))

    def is_finite(self) -> Self:
        return self._from_ir(boolean.IsFinite().to_function_expr(self._ir))

    def is_nan(self) -> Self:
        return self._from_ir(boolean.IsNan().to_function_expr(self._ir))

    def is_null(self) -> Self:
        return self._from_ir(boolean.IsNull().to_function_expr(self._ir))

    def is_first_distinct(self) -> Self:
        return self._from_ir(boolean.IsFirstDistinct().to_function_expr(self._ir))

    def is_last_distinct(self) -> Self:
        return self._from_ir(boolean.IsLastDistinct().to_function_expr(self._ir))

    def is_unique(self) -> Self:
        return self._from_ir(boolean.IsUnique().to_function_expr(self._ir))

    def is_between(
        self,
        lower_bound: IntoExpr,
        upper_bound: IntoExpr,
        closed: ClosedInterval = "both",
    ) -> Self:
        it = parse.parse_into_seq_of_expr_ir(lower_bound, upper_bound)
        return self._from_ir(
            boolean.IsBetween(closed=closed).to_function_expr(self._ir, *it)
        )

    def is_in(self, other: t.Iterable[t.Any]) -> Self:
        node: boolean.IsIn[t.Any]
        if is_series(other):
            node = boolean.IsInSeries.from_series(other)
        elif isinstance(other, t.Iterable):
            node = boolean.IsInSeq.from_iterable(other)
        elif is_expr(other):
            node = boolean.IsInExpr(other=other._ir)
        else:
            msg = f"`is_in` only supports iterables, got: {type(other).__name__}"
            raise TypeError(msg)
        return self._from_ir(node.to_function_expr(self._ir))

    def __eq__(self, other: IntoExpr) -> Self:  # type: ignore[override]
        op = ops.Eq()
        rhs = parse.parse_into_expr_ir(other, str_as_lit=True)
        return self._from_ir(op.to_binary_expr(self._ir, rhs))

    def __ne__(self, other: IntoExpr) -> Self:  # type: ignore[override]
        op = ops.NotEq()
        rhs = parse.parse_into_expr_ir(other, str_as_lit=True)
        return self._from_ir(op.to_binary_expr(self._ir, rhs))

    def __lt__(self, other: IntoExpr) -> Self:
        op = ops.Lt()
        rhs = parse.parse_into_expr_ir(other, str_as_lit=True)
        return self._from_ir(op.to_binary_expr(self._ir, rhs))

    def __le__(self, other: IntoExpr) -> Self:
        op = ops.LtEq()
        rhs = parse.parse_into_expr_ir(other, str_as_lit=True)
        return self._from_ir(op.to_binary_expr(self._ir, rhs))

    def __gt__(self, other: IntoExpr) -> Self:
        op = ops.Gt()
        rhs = parse.parse_into_expr_ir(other, str_as_lit=True)
        return self._from_ir(op.to_binary_expr(self._ir, rhs))

    def __ge__(self, other: IntoExpr) -> Self:
        op = ops.GtEq()
        rhs = parse.parse_into_expr_ir(other, str_as_lit=True)
        return self._from_ir(op.to_binary_expr(self._ir, rhs))

    def __add__(self, other: IntoExpr) -> Self:
        op = ops.Add()
        rhs = parse.parse_into_expr_ir(other, str_as_lit=True)
        return self._from_ir(op.to_binary_expr(self._ir, rhs))

    def __sub__(self, other: IntoExpr) -> Self:
        op = ops.Sub()
        rhs = parse.parse_into_expr_ir(other, str_as_lit=True)
        return self._from_ir(op.to_binary_expr(self._ir, rhs))

    def __mul__(self, other: IntoExpr) -> Self:
        op = ops.Multiply()
        rhs = parse.parse_into_expr_ir(other, str_as_lit=True)
        return self._from_ir(op.to_binary_expr(self._ir, rhs))

    def __truediv__(self, other: IntoExpr) -> Self:
        op = ops.TrueDivide()
        rhs = parse.parse_into_expr_ir(other, str_as_lit=True)
        return self._from_ir(op.to_binary_expr(self._ir, rhs))

    def __floordiv__(self, other: IntoExpr) -> Self:
        op = ops.FloorDivide()
        rhs = parse.parse_into_expr_ir(other, str_as_lit=True)
        return self._from_ir(op.to_binary_expr(self._ir, rhs))

    def __mod__(self, other: IntoExpr) -> Self:
        op = ops.Modulus()
        rhs = parse.parse_into_expr_ir(other, str_as_lit=True)
        return self._from_ir(op.to_binary_expr(self._ir, rhs))

    def __and__(self, other: IntoExpr) -> Self:
        op = ops.And()
        rhs = parse.parse_into_expr_ir(other, str_as_lit=True)
        return self._from_ir(op.to_binary_expr(self._ir, rhs))

    def __rand__(self, other: IntoExpr) -> Self:
        return (self & other).alias("literal")

    def __or__(self, other: IntoExpr) -> Self:
        op = ops.Or()
        rhs = parse.parse_into_expr_ir(other, str_as_lit=True)
        return self._from_ir(op.to_binary_expr(self._ir, rhs))

    def __ror__(self, other: IntoExpr) -> Self:
        return (self | other).alias("literal")

    def __xor__(self, other: IntoExpr) -> Self:
        op = ops.ExclusiveOr()
        rhs = parse.parse_into_expr_ir(other, str_as_lit=True)
        return self._from_ir(op.to_binary_expr(self._ir, rhs))

    def __rxor__(self, other: IntoExpr) -> Self:
        return (self ^ other).alias("literal")

    def __invert__(self) -> Self:
        return self._from_ir(boolean.Not().to_function_expr(self._ir))

    def __pow__(self, other: IntoExpr) -> Self:
        exponent = parse.parse_into_expr_ir(other, str_as_lit=True)
        base = self._ir
        return self._from_ir(F.Pow().to_function_expr(base, exponent))

    @property
    def meta(self) -> IRMetaNamespace:
        from narwhals._plan.meta import IRMetaNamespace

        return IRMetaNamespace.from_expr(self)

    @property
    def name(self) -> ExprNameNamespace:
        """Specialized expressions for modifying the name of existing expressions.

        Examples:
            >>> from narwhals._plan import demo as nw
            >>>
            >>> renamed = nw.col("a", "b").name.suffix("_changed")
            >>> str(renamed._ir)
            "RenameAlias(expr=Columns(names=[a, b]), function=Suffix(suffix='_changed'))"
        """
        from narwhals._plan.name import ExprNameNamespace

        return ExprNameNamespace(_expr=self)

    @property
    def cat(self) -> ExprCatNamespace:
        from narwhals._plan.categorical import ExprCatNamespace

        return ExprCatNamespace(_expr=self)

    @property
    def struct(self) -> ExprStructNamespace:
        from narwhals._plan.struct import ExprStructNamespace

        return ExprStructNamespace(_expr=self)

    @property
    def dt(self) -> ExprDateTimeNamespace:
        from narwhals._plan.temporal import ExprDateTimeNamespace

        return ExprDateTimeNamespace(_expr=self)

    @property
    def list(self) -> ExprListNamespace:
        from narwhals._plan.lists import ExprListNamespace

        return ExprListNamespace(_expr=self)

    @property
    def str(self) -> ExprStringNamespace:
        from narwhals._plan.strings import ExprStringNamespace

        return ExprStringNamespace(_expr=self)


class DummySelector(DummyExpr):
    """Selectors placeholder.

    Examples:
        >>> from narwhals._plan import selectors as ncs
        >>>
        >>> (ncs.matches("[^z]a") & ncs.string()) | ncs.datetime("us", None)
        Narwhals DummySelector (main):
        [([(ncs.matches(pattern='[^z]a')) & (ncs.string())]) | (ncs.datetime(time_unit=['us'], time_zone=[None]))]
        >>>
        >>> ~(ncs.boolean() | ncs.matches(r"is_.*"))
        Narwhals DummySelector (main):
        ~[(ncs.boolean()) | (ncs.matches(pattern='is_.*'))]
    """

    _ir: expr.SelectorIR

    def __repr__(self) -> str:
        return f"Narwhals DummySelector ({self.version.name.lower()}):\n{self._ir!r}"

    @classmethod
    def _from_ir(cls, ir: expr.SelectorIR, /) -> Self:  # type: ignore[override]
        obj = cls.__new__(cls)
        obj._ir = ir
        return obj

    def _to_expr(self) -> DummyExpr:
        return self._ir.to_narwhals(self.version)

    @t.overload  # type: ignore[override]
    def __or__(self, other: Self) -> Self: ...
    @t.overload
    def __or__(self, other: IntoExpr) -> DummyExpr: ...
    def __or__(self, other: IntoExpr) -> Self | DummyExpr:
        if isinstance(other, type(self)):
            op = ops.Or()
            return self._from_ir(op.to_binary_selector(self._ir, other._ir))
        return self._to_expr() | other

    @t.overload  # type: ignore[override]
    def __and__(self, other: Self) -> Self: ...
    @t.overload
    def __and__(self, other: IntoExpr) -> DummyExpr: ...
    def __and__(self, other: IntoExpr) -> Self | DummyExpr:
        if is_column(other) and (name := other.meta.output_name()):
            other = by_name(name)
        if isinstance(other, type(self)):
            op = ops.And()
            return self._from_ir(op.to_binary_selector(self._ir, other._ir))
        return self._to_expr() & other

    @t.overload  # type: ignore[override]
    def __sub__(self, other: Self) -> Self: ...
    @t.overload
    def __sub__(self, other: IntoExpr) -> DummyExpr: ...
    def __sub__(self, other: IntoExpr) -> Self | DummyExpr:
        if isinstance(other, type(self)):
            op = ops.Sub()
            return self._from_ir(op.to_binary_selector(self._ir, other._ir))
        return self._to_expr() - other

    @t.overload  # type: ignore[override]
    def __xor__(self, other: Self) -> Self: ...
    @t.overload
    def __xor__(self, other: IntoExpr) -> DummyExpr: ...
    def __xor__(self, other: IntoExpr) -> Self | DummyExpr:
        if isinstance(other, type(self)):
            op = ops.ExclusiveOr()
            return self._from_ir(op.to_binary_selector(self._ir, other._ir))
        return self._to_expr() ^ other

    def __invert__(self) -> Self:
        return self._from_ir(expr.InvertSelector(selector=self._ir))

    def __add__(self, other: t.Any) -> DummyExpr:  # type: ignore[override]
        if isinstance(other, type(self)):
            msg = "unsupported operand type(s) for op: ('Selector' + 'Selector')"
            raise TypeError(msg)
        return self._to_expr() + other  # type: ignore[no-any-return]

    def __radd__(self, other: t.Any) -> Never:
        msg = "unsupported operand type(s) for op: ('Expr' + 'Selector')"
        raise TypeError(msg)

    def __rsub__(self, other: t.Any) -> Never:
        msg = "unsupported operand type(s) for op: ('Expr' - 'Selector')"
        raise TypeError(msg)

    @t.overload  # type: ignore[override]
    def __rand__(self, other: Self) -> Self: ...
    @t.overload
    def __rand__(self, other: IntoExpr) -> DummyExpr: ...
    def __rand__(self, other: IntoExpr) -> Self | DummyExpr:
        if is_column(other) and (name := other.meta.output_name()):
            return by_name(name) & self
        return self._to_expr().__rand__(other)

    @t.overload  # type: ignore[override]
    def __ror__(self, other: Self) -> Self: ...
    @t.overload
    def __ror__(self, other: IntoExpr) -> DummyExpr: ...
    def __ror__(self, other: IntoExpr) -> Self | DummyExpr:
        if is_column(other) and (name := other.meta.output_name()):
            return by_name(name) | self
        return self._to_expr().__ror__(other)

    @t.overload  # type: ignore[override]
    def __rxor__(self, other: Self) -> Self: ...
    @t.overload
    def __rxor__(self, other: IntoExpr) -> DummyExpr: ...
    def __rxor__(self, other: IntoExpr) -> Self | DummyExpr:
        if is_column(other) and (name := other.meta.output_name()):
            return by_name(name) ^ self
        return self._to_expr().__rxor__(other)


class DummyExprV1(DummyExpr):
    _version: t.ClassVar[Version] = Version.V1


class DummySelectorV1(DummySelector):
    _version: t.ClassVar[Version] = Version.V1


class DummyCompliantExpr:
    _ir: ExprIR
    _version: Version

    @property
    def version(self) -> Version:
        return self._version

    @classmethod
    def _from_ir(cls, ir: ExprIR, /, version: Version) -> Self:
        obj = cls.__new__(cls)
        obj._ir = ir
        obj._version = version
        return obj

    def to_narwhals(self) -> DummyExpr:
        if self.version is Version.MAIN:
            return DummyExpr._from_ir(self._ir)
        return DummyExprV1._from_ir(self._ir)


class DummyFrame(Generic[NativeFrameT, NativeSeriesT]):
    _compliant: CompliantFrame[NativeFrameT, NativeSeriesT]
    _version: t.ClassVar[Version] = Version.MAIN

    @property
    def version(self) -> Version:
        return self._version

    @property
    def _series(self) -> type[DummySeries[NativeSeriesT]]:
        return DummySeries[NativeSeriesT]

    @property
    def schema(self) -> Schema:
        return Schema(self._compliant.schema.items())

    @property
    def columns(self) -> list[str]:
        return self._compliant.columns

    # NOTE: Gave up on trying to get typing working for now
    @classmethod
    def from_native(
        cls, native: NativeFrame, /
    ) -> DummyFrame[pa.Table, pa.ChunkedArray[t.Any]]:
        if is_pyarrow_table(native):
            from narwhals._plan.arrow.dataframe import ArrowDataFrame

            return ArrowDataFrame.from_native(native, cls._version).to_narwhals()

        raise NotImplementedError(type(native))

    @classmethod
    def _from_compliant(
        cls, compliant: CompliantFrame[NativeFrameT, NativeSeriesT], /
    ) -> Self:
        obj = cls.__new__(cls)
        obj._compliant = compliant
        return obj

    def to_native(self) -> NativeFrameT:
        return self._compliant.native

    @t.overload
    def to_dict(
        self, *, as_series: t.Literal[True] = ...
    ) -> dict[str, DummySeries[NativeSeriesT]]: ...

    @t.overload
    def to_dict(self, *, as_series: t.Literal[False]) -> dict[str, list[t.Any]]: ...

    @t.overload
    def to_dict(
        self, *, as_series: bool
    ) -> dict[str, DummySeries[NativeSeriesT]] | dict[str, list[t.Any]]: ...

    def to_dict(
        self, *, as_series: bool = True
    ) -> dict[str, DummySeries[NativeSeriesT]] | dict[str, list[t.Any]]:
        if as_series:
            return {
                key: self._series._from_compliant(value)
                for key, value in self._compliant.to_dict(as_series=as_series).items()
            }
        return self._compliant.to_dict(as_series=as_series)

    def __len__(self) -> int:
        return len(self._compliant)

    def _project(
        self,
        exprs: tuple[IntoExpr | Iterable[IntoExpr], ...],
        named_exprs: dict[str, t.Any],
        context: ExprContext,
        /,
    ) -> tuple[Seq[NamedIR[ExprIR]], FrozenSchema]:
        """Temp, while these parts aren't connected, this is easier for testing."""
        irs, schema_frozen, output_names = expr_expansion.prepare_projection(
            parse.parse_into_seq_of_expr_ir(*exprs, **named_exprs), self.schema
        )
        named_irs = expr_expansion.into_named_irs(irs, output_names)
        return schema_frozen.project(named_irs, context)

    def select(self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: t.Any) -> Self:
        named_irs, schema_projected = self._project(
            exprs, named_exprs, ExprContext.SELECT
        )
        return self._from_compliant(self._compliant.select(named_irs, schema_projected))

    def sort(
        self,
        by: str | Iterable[str],
        *more_by: str,
        descending: bool | Sequence[bool] = False,
        nulls_last: bool | Sequence[bool] = False,
    ) -> Self:
        sort, opts = _parse_sort_by(
            by, *more_by, descending=descending, nulls_last=nulls_last
        )
        irs, schema_frozen, output_names = expr_expansion.prepare_projection(
            sort, self.schema
        )
        named_irs = expr_expansion.into_named_irs(irs, output_names)
        return self._from_compliant(self._compliant.sort(named_irs, opts, schema_frozen))


class DummySeries(Generic[NativeSeriesT]):
    _compliant: DummyCompliantSeries[NativeSeriesT]
    _version: t.ClassVar[Version] = Version.MAIN

    @property
    def version(self) -> Version:
        return self._version

    @property
    def dtype(self) -> DType:
        return self._compliant.dtype

    @property
    def name(self) -> str:
        return self._compliant.name

    # NOTE: Gave up on trying to get typing working for now
    @classmethod
    def from_native(
        cls, native: NativeSeries, name: str = "", /
    ) -> DummySeries[pa.ChunkedArray[t.Any]]:
        if is_pyarrow_chunked_array(native):
            from narwhals._plan.arrow.series import ArrowSeries

            return ArrowSeries.from_native(
                native, name, version=cls._version
            ).to_narwhals()

        raise NotImplementedError(type(native))

    @classmethod
    def _from_compliant(cls, compliant: DummyCompliantSeries[NativeSeriesT], /) -> Self:
        obj = cls.__new__(cls)
        obj._compliant = compliant
        return obj

    def to_native(self) -> NativeSeriesT:
        return self._compliant.native

    def to_list(self) -> list[t.Any]:
        return self._compliant.to_list()

    def __iter__(self) -> t.Iterator[t.Any]:
        yield from self.to_native()


class DummySeriesV1(DummySeries[NativeSeriesT]):
    _version: t.ClassVar[Version] = Version.V1
