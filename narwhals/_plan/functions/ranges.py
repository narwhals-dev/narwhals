from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any, Literal, overload

from narwhals._duration import Interval
from narwhals._plan import _parse, common
from narwhals._plan._dispatch import get_dispatch_name
from narwhals._plan._namespace import eager_namespace
from narwhals._plan.expressions.ranges import (
    DateRange,
    IntRange,
    LinearSpace,
    RangeFunction,
)
from narwhals._utils import Version, ensure_type, qualified_type_name
from narwhals.exceptions import ComputeError, InvalidOperationError

if TYPE_CHECKING:
    import pyarrow as pa

    from narwhals._plan._namespace import EagerNs
    from narwhals._plan.expr import Expr
    from narwhals._plan.series import Series
    from narwhals._plan.typing import IntoExprColumn, NativeSeriesT, NonNestedLiteralT
    from narwhals._typing import Arrow
    from narwhals.dtypes import IntegerType
    from narwhals.typing import ClosedInterval, EagerAllowed, IntoBackend


@overload
def int_range(
    start: int | IntoExprColumn = ...,
    end: int | IntoExprColumn | None = ...,
    step: int = ...,
    *,
    dtype: IntegerType | type[IntegerType] = ...,
    eager: Literal[False] = ...,
) -> Expr: ...
@overload
def int_range(
    start: int = ...,
    end: int | None = ...,
    step: int = ...,
    *,
    dtype: IntegerType | type[IntegerType] = ...,
    eager: Arrow,
) -> Series[pa.ChunkedArray[Any]]: ...
@overload
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
    eager: IntoBackend[EagerAllowed] | Literal[False] = False,
) -> Expr | Series:
    if end is None:
        end = start
        start = 0
    dtype = common.into_dtype(dtype)
    if eager:
        ns = eager_namespace(eager)
        start, end = _ensure_range_scalar(start, end, int, IntRange, eager)
        return _int_range_eager(start, end, step, dtype, ns)
    return (
        IntRange(step=step, dtype=dtype)
        .to_function_expr(*_parse.parse_into_seq_of_expr_ir(start, end))
        .to_narwhals()
    )


@overload
def date_range(
    start: dt.date | IntoExprColumn,
    end: dt.date | IntoExprColumn,
    interval: str | dt.timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    eager: Literal[False] = ...,
) -> Expr: ...
@overload
def date_range(
    start: dt.date,
    end: dt.date,
    interval: str | dt.timedelta = ...,
    *,
    closed: ClosedInterval = ...,
    eager: Arrow,
) -> Series[pa.ChunkedArray[Any]]: ...
@overload
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
    eager: IntoBackend[EagerAllowed] | Literal[False] = False,
) -> Expr | Series:
    days = _interval_days(interval)
    closed = _ensure_closed_interval(closed)
    if eager:
        ns = eager_namespace(eager)
        start, end = _ensure_range_scalar(start, end, dt.date, DateRange, eager)
        return _date_range_eager(start, end, days, closed, ns)
    return (
        DateRange(interval=days, closed=closed)
        .to_function_expr(*_parse.parse_into_seq_of_expr_ir(start, end))
        .to_narwhals()
    )


@overload
def linear_space(
    start: float | IntoExprColumn,
    end: float | IntoExprColumn,
    num_samples: int,
    *,
    closed: ClosedInterval = ...,
    eager: Literal[False] = ...,
) -> Expr: ...
@overload
def linear_space(
    start: float,
    end: float,
    num_samples: int,
    *,
    closed: ClosedInterval = ...,
    eager: Arrow,
) -> Series[pa.ChunkedArray[Any]]: ...
@overload
def linear_space(
    start: float,
    end: float,
    num_samples: int,
    *,
    closed: ClosedInterval = ...,
    eager: IntoBackend[EagerAllowed],
) -> Series: ...
def linear_space(
    start: float | IntoExprColumn,
    end: float | IntoExprColumn,
    num_samples: int,
    *,
    closed: ClosedInterval = "both",
    eager: IntoBackend[EagerAllowed] | Literal[False] = False,
) -> Expr | Series:
    """Create sequence of evenly-spaced points.

    Arguments:
        start: Lower bound of the range.
        end: Upper bound of the range.
        num_samples: Number of samples in the output sequence.
        closed: Define which sides of the interval are closed (inclusive).
        eager: If set to `False` (default), then an expression is returned.
            If set to an (eager) implementation ("pandas", "polars" or "pyarrow"), then
            a `Series` is returned.

    Notes:
        Unlike `pl.linear_space`, *currently* only numeric dtypes (and not temporal) are supported.

    Examples:
        >>> import narwhals._plan as nwp
        >>> nwp.linear_space(start=0, end=1, num_samples=3, eager="pyarrow").to_list()
        [0.0, 0.5, 1.0]

        >>> nwp.linear_space(0, 1, 3, closed="left", eager="pyarrow").to_list()
        [0.0, 0.3333333333333333, 0.6666666666666666]

        >>> nwp.linear_space(0, 1, 3, closed="right", eager="pyarrow").to_list()
        [0.3333333333333333, 0.6666666666666666, 1.0]

        >>> nwp.linear_space(0, 1, 3, closed="none", eager="pyarrow").to_list()
        [0.25, 0.5, 0.75]

        >>> df = nwp.DataFrame.from_dict({"a": [1, 2, 3, 4, 5]}, backend="pyarrow")
        >>> df.with_columns(nwp.linear_space(0, 10, 5).alias("ls"))
        ┌──────────────────────┐
        |     nw.DataFrame     |
        |----------------------|
        |pyarrow.Table         |
        |a: int64              |
        |ls: double            |
        |----                  |
        |a: [[1,2,3,4,5]]      |
        |ls: [[0,2.5,5,7.5,10]]|
        └──────────────────────┘
    """
    ensure_type(num_samples, int, param_name="num_samples")
    closed = _ensure_closed_interval(closed)
    if eager:
        ns = eager_namespace(eager)
        start, end = _ensure_range_scalar(start, end, (float, int), LinearSpace, eager)
        return _linear_space_eager(start, end, num_samples, closed, ns)
    return (
        LinearSpace(num_samples=num_samples, closed=closed)
        .to_function_expr(*_parse.parse_into_seq_of_expr_ir(start, end))
        .to_narwhals()
    )


# TODO @dangotbanned: Handle this in `RangeFunction` or `RangeExpr`
# NOTE: `ArrowNamespace._range_function_inputs` has some duplicated logic too
def _ensure_range_scalar(
    start: Any,
    end: Any,
    valid_type: type[NonNestedLiteralT] | tuple[type[NonNestedLiteralT], ...],
    function: type[RangeFunction],
    eager: IntoBackend[EagerAllowed],
) -> tuple[NonNestedLiteralT, NonNestedLiteralT]:
    if isinstance(start, valid_type) and isinstance(end, valid_type):
        return start, end
    tp_start = qualified_type_name(start)
    tp_end = qualified_type_name(end)
    valid_types = (valid_type,) if not isinstance(valid_type, tuple) else valid_type
    tp_names = " | ".join(tp.__name__ for tp in valid_types)
    msg = (
        f"Expected `start` and `end` to be {tp_names} values since `eager={eager}`, but got: ({tp_start}, {tp_end})\n\n"
        f"Hint: Calling `nw.{get_dispatch_name(function)}` with expressions requires:\n"
        "  - `eager=False`\n"
        "  - a context such as `select` or `with_columns`"
    )
    raise InvalidOperationError(msg)


# NOTE: The extra level of indirection here is to satisfy `mypy`
# AFAICT, `pyright` is able to use bidirectional inference from the `{date,int}_range` overloads.
# `mypy` would treat the same call as an `Any`
def _date_range_eager(
    start: dt.date,
    end: dt.date,
    interval: int,
    closed: ClosedInterval,
    ns: EagerNs[Any, NativeSeriesT],
    /,
) -> Series[NativeSeriesT]:
    return ns.date_range_eager(start, end, interval, closed=closed).to_narwhals()


# NOTE: See `_date_range_eager`
def _int_range_eager(
    start: int,
    end: int,
    step: int,
    dtype: IntegerType,
    ns: EagerNs[Any, NativeSeriesT],
    /,
) -> Series[NativeSeriesT]:
    return ns.int_range_eager(start, end, step, dtype=dtype).to_narwhals()


def _linear_space_eager(
    start: float,
    end: float,
    num_samples: int,
    closed: ClosedInterval,
    ns: EagerNs[Any, NativeSeriesT],
    /,
) -> Series[NativeSeriesT]:
    return ns.linear_space_eager(start, end, num_samples, closed=closed).to_narwhals()


def _ensure_closed_interval(closed: ClosedInterval, /) -> ClosedInterval:
    closed_intervals = "left", "right", "none", "both"
    if closed not in closed_intervals:
        msg = f"`closed` must be one of {closed_intervals!r}, got {closed!r}"
        raise TypeError(msg)
    return closed


def _interval_days(interval: str | dt.timedelta, /) -> int:
    if not isinstance(interval, dt.timedelta):
        if interval == "1d":
            return 1
        parsed = Interval.parse_no_constraints(interval)
        if parsed.unit not in {"d", "w", "mo", "q", "y"}:
            msg = f"`interval` input for `date_range` must consist of full days, got: {parsed.multiple}{parsed.unit}"
            raise ComputeError(msg)
        if parsed.unit in {"mo", "q", "y"}:
            msg = f"`interval` input for `date_range` does not support {parsed.unit!r} yet, got: {parsed.multiple}{parsed.unit}"
            raise NotImplementedError(msg)
        interval = parsed.to_timedelta()
    return interval.days
