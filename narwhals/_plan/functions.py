from __future__ import annotations

import builtins
import datetime as dt
import typing as t
from typing import TYPE_CHECKING

from narwhals._duration import Interval
from narwhals._plan import _guards, _parse, common, expressions as ir, selectors as cs
from narwhals._plan.expressions import functions as F
from narwhals._plan.expressions.literal import ScalarLiteral, SeriesLiteral
from narwhals._plan.expressions.ranges import DateRange, IntRange
from narwhals._plan.expressions.strings import ConcatStr
from narwhals._plan.when_then import When
from narwhals._utils import Implementation, Version, flatten, is_eager_allowed
from narwhals.exceptions import ComputeError, InvalidOperationError

if TYPE_CHECKING:
    import pyarrow as pa

    from narwhals._plan import arrow as _arrow
    from narwhals._plan.compliant.namespace import EagerNamespace
    from narwhals._plan.compliant.series import CompliantSeries
    from narwhals._plan.expr import Expr
    from narwhals._plan.series import Series
    from narwhals._plan.typing import IntoExpr, IntoExprColumn, NativeSeriesT
    from narwhals._typing import Arrow
    from narwhals.dtypes import IntegerType
    from narwhals.typing import (
        ClosedInterval,
        EagerAllowed,
        IntoBackend,
        IntoDType,
        NonNestedLiteral,
    )


def col(*names: str | t.Iterable[str]) -> Expr:
    flat = tuple(flatten(names))
    return (
        ir.col(flat[0]).to_narwhals()
        if builtins.len(flat) == 1
        else cs.by_name(*flat).as_expr()
    )


def nth(*indices: int | t.Sequence[int]) -> Expr:
    return cs.by_index(*indices).as_expr()


def lit(
    value: NonNestedLiteral | Series[NativeSeriesT], dtype: IntoDType | None = None
) -> Expr:
    if _guards.is_series(value):
        return SeriesLiteral(value=value).to_literal().to_narwhals()
    if not _guards.is_non_nested_literal(value):
        msg = f"{type(value).__name__!r} is not supported in `nw.lit`, got: {value!r}."
        raise TypeError(msg)
    if dtype is None:
        dtype = common.py_to_narwhals_dtype(value, Version.MAIN)
    else:
        dtype = common.into_dtype(dtype)
    return ScalarLiteral(value=value, dtype=dtype).to_literal().to_narwhals()


def len() -> Expr:
    return ir.Len().to_narwhals()


def all() -> Expr:
    return cs.all().as_expr()


def exclude(*names: str | t.Iterable[str]) -> Expr:
    return cs.all().exclude(*names).as_expr()


def max(*columns: str) -> Expr:
    return col(columns).max()


def mean(*columns: str) -> Expr:
    return col(columns).mean()


def min(*columns: str) -> Expr:
    return col(columns).min()


def median(*columns: str) -> Expr:
    return col(columns).median()


def sum(*columns: str) -> Expr:
    return col(columns).sum()


def all_horizontal(*exprs: IntoExpr | t.Iterable[IntoExpr]) -> Expr:
    it = _parse.parse_into_seq_of_expr_ir(*exprs)
    return ir.boolean.AllHorizontal().to_function_expr(*it).to_narwhals()


def any_horizontal(*exprs: IntoExpr | t.Iterable[IntoExpr]) -> Expr:
    it = _parse.parse_into_seq_of_expr_ir(*exprs)
    return ir.boolean.AnyHorizontal().to_function_expr(*it).to_narwhals()


def sum_horizontal(*exprs: IntoExpr | t.Iterable[IntoExpr]) -> Expr:
    it = _parse.parse_into_seq_of_expr_ir(*exprs)
    return F.SumHorizontal().to_function_expr(*it).to_narwhals()


def min_horizontal(*exprs: IntoExpr | t.Iterable[IntoExpr]) -> Expr:
    it = _parse.parse_into_seq_of_expr_ir(*exprs)
    return F.MinHorizontal().to_function_expr(*it).to_narwhals()


def max_horizontal(*exprs: IntoExpr | t.Iterable[IntoExpr]) -> Expr:
    it = _parse.parse_into_seq_of_expr_ir(*exprs)
    return F.MaxHorizontal().to_function_expr(*it).to_narwhals()


def mean_horizontal(*exprs: IntoExpr | t.Iterable[IntoExpr]) -> Expr:
    it = _parse.parse_into_seq_of_expr_ir(*exprs)
    return F.MeanHorizontal().to_function_expr(*it).to_narwhals()


def concat_str(
    exprs: IntoExpr | t.Iterable[IntoExpr],
    *more_exprs: IntoExpr,
    separator: str = "",
    ignore_nulls: bool = False,
) -> Expr:
    it = _parse.parse_into_seq_of_expr_ir(exprs, *more_exprs)
    return (
        ConcatStr(separator=separator, ignore_nulls=ignore_nulls)
        .to_function_expr(*it)
        .to_narwhals()
    )


def when(
    *predicates: IntoExprColumn | t.Iterable[IntoExprColumn], **constraints: t.Any
) -> When:
    """Start a `when-then-otherwise` expression.

    Examples:
        >>> from narwhals import _plan as nw

        >>> nw.when(nw.col("y") == "b").then(1)
        nw._plan.Expr(main):
        .when([(col('y')) == (lit(str: b))]).then(lit(int: 1)).otherwise(lit(null))
    """
    condition = _parse.parse_predicates_constraints_into_expr_ir(
        *predicates, **constraints
    )
    return When._from_ir(condition)


@t.overload
def int_range(
    start: int | IntoExprColumn = ...,
    end: int | IntoExprColumn | None = ...,
    step: int = ...,
    *,
    dtype: IntegerType | type[IntegerType] = ...,
    eager: t.Literal[False] = ...,
) -> Expr: ...
@t.overload
def int_range(
    start: int = ...,
    end: int | None = ...,
    step: int = ...,
    *,
    dtype: IntegerType | type[IntegerType] = ...,
    eager: Arrow,
) -> Series[pa.ChunkedArray[t.Any]]: ...
@t.overload
def int_range(
    start: int = ...,
    end: int | None = ...,
    step: int = ...,
    *,
    dtype: IntegerType | type[IntegerType] = ...,
    eager: IntoBackend[EagerAllowed],
) -> Series: ...
def int_range(
    start: int | IntoExprColumn = 0,
    end: int | IntoExprColumn | None = None,
    step: int = 1,
    *,
    dtype: IntegerType | type[IntegerType] = Version.MAIN.dtypes.Int64,
    eager: IntoBackend[EagerAllowed] | t.Literal[False] = False,
) -> Expr | Series:
    if end is None:
        end = start
        start = 0
    dtype = common.into_dtype(dtype)
    if eager:
        return _int_range_eager(start, end, step, dtype=dtype, ns=_eager_namespace(eager))
    return (
        IntRange(step=step, dtype=dtype)
        .to_function_expr(*_parse.parse_into_seq_of_expr_ir(start, end))
        .to_narwhals()
    )


# TODO @dangotbanned: Deduplicate `_{int,date}_range_eager`
def _int_range_eager(
    start: t.Any,
    end: t.Any,
    step: int,
    *,
    dtype: IntegerType,
    ns: EagerNamespace[t.Any, CompliantSeries[NativeSeriesT], t.Any, t.Any],
) -> Series[NativeSeriesT]:
    if not (isinstance(start, int) and isinstance(end, int)):
        msg = (
            f"Expected `start` and `end` to be integer values since `eager={ns.implementation}`.\n"
            f"Found: `start` of type {type(start)} and `end` of type {type(end)}\n\n"
            "Hint: Calling `nw.int_range` with expressions requires:\n"
            "  - `eager=False`"
            "  - a context such as `select` or `with_columns`"
        )
        raise InvalidOperationError(msg)
    return ns.int_range_eager(start, end, step, dtype=dtype).to_narwhals()


@t.overload
def _eager_namespace(backend: Arrow, /) -> _arrow.Namespace: ...
@t.overload
def _eager_namespace(
    backend: IntoBackend[EagerAllowed], /
) -> EagerNamespace[t.Any, t.Any, t.Any, t.Any]: ...
def _eager_namespace(
    backend: IntoBackend[EagerAllowed], /
) -> EagerNamespace[t.Any, t.Any, t.Any, t.Any] | _arrow.Namespace:
    impl = Implementation.from_backend(backend)
    if is_eager_allowed(impl):
        if impl is Implementation.PYARROW:
            from narwhals._plan.arrow.namespace import ArrowNamespace

            return ArrowNamespace(Version.MAIN)
        raise NotImplementedError(impl)
    msg = f"{impl} support in Narwhals is lazy-only"
    raise ValueError(msg)


@t.overload
def date_range(
    start: dt.date | IntoExprColumn,
    end: dt.date | IntoExprColumn,
    interval: str | dt.timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    eager: t.Literal[False] = ...,
) -> Expr: ...
@t.overload
def date_range(
    start: dt.date,
    end: dt.date,
    interval: str | dt.timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    eager: Arrow,
) -> Series[pa.ChunkedArray[t.Any]]: ...
@t.overload
def date_range(
    start: dt.date,
    end: dt.date,
    interval: str | dt.timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    eager: IntoBackend[EagerAllowed],
) -> Series: ...
def date_range(
    start: dt.date | IntoExprColumn,
    end: dt.date | IntoExprColumn,
    interval: str | dt.timedelta = "1d",
    *,
    closed: ClosedInterval = "both",
    eager: IntoBackend[EagerAllowed] | t.Literal[False] = False,
) -> Expr | Series:
    days = _interval_days(interval)
    closed = _ensure_closed_interval(closed)
    if eager:
        ns = _eager_namespace(eager)
        return _date_range_eager(start, end, days, closed=closed, ns=ns)
    return (
        DateRange(interval=days, closed=closed)
        .to_function_expr(*_parse.parse_into_seq_of_expr_ir(start, end))
        .to_narwhals()
    )


def _ensure_closed_interval(closed: ClosedInterval, /) -> ClosedInterval:
    closed_intervals = "left", "right", "none", "both"
    if closed not in closed_intervals:
        msg = f"`closed` must be one of {closed_intervals!r}, got {closed!r}"
        raise TypeError(msg)
    return closed


# TODO @dangotbanned: Deduplicate `_{int,date}_range_eager`
def _date_range_eager(
    start: t.Any,
    end: t.Any,
    interval: int,
    *,
    closed: ClosedInterval,
    ns: EagerNamespace[t.Any, CompliantSeries[NativeSeriesT], t.Any, t.Any],
) -> Series[NativeSeriesT]:
    if not (isinstance(start, dt.date) and isinstance(end, dt.date)):
        msg = (
            f"Expected `start` and `end` to be date values since `eager={ns.implementation}`.\n"
            f"Found: `start` of type {type(start)} and `end` of type {type(end)}\n\n"
            "Hint: Calling `nw.date_range` with expressions requires:\n"
            "  - `eager=False`"
            "  - a context such as `select` or `with_columns`"
        )
        raise InvalidOperationError(msg)
    return ns.date_range_eager(start, end, interval, closed=closed).to_narwhals()


def _interval_days(interval: str | dt.timedelta, /) -> int:
    if not isinstance(interval, dt.timedelta):
        if interval == "1d":
            return 1
        parsed = Interval.parse_no_constraints(interval)
        if parsed.unit not in {"d", "mo", "q", "y"}:
            msg = f"`interval` input for `date_range` must consist of full days, got: {parsed.multiple}{parsed.unit}"
            raise ComputeError(msg)
        if parsed.unit in {"mo", "q", "y"}:
            msg = f"`interval` input for `date_range` does not support {parsed.unit!r} yet, got: {parsed.multiple}{parsed.unit}"
            raise NotImplementedError(msg)
        interval = parsed.to_timedelta()
    return interval.days
