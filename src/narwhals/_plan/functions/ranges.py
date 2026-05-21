from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING, Any, Literal, overload

from narwhals._duration import Interval
from narwhals._plan import _parse, common, plugins
from narwhals._plan.expressions.ranges import DateRange, IntRange, LinearSpace
from narwhals._utils import Version, ensure_type
from narwhals.exceptions import ComputeError

if TYPE_CHECKING:
    import polars as pl
    import pyarrow as pa

    from narwhals._plan.expr import Expr
    from narwhals._plan.series import Series
    from narwhals._plan.typing import IntoExprColumn
    from narwhals._typing import Arrow, Polars
    from narwhals.dtypes import IntegerType
    from narwhals.typing import ClosedInterval, EagerAllowed, IntoBackend

_MAIN = Version.MAIN


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
    eager: Polars,
) -> Series[pl.Series]: ...
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
        start, end = IntRange.ensure_py_scalars(start, end, eager)
        return (
            plugins.manager()
            .series(eager, _MAIN)
            .int_range(start, end, step, dtype=dtype)
            .to_narwhals()
        )
    return (
        IntRange(step=step, dtype=dtype)
        .to_function_expr(*_parse.into_iter_expr_ir(start, end))
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
    eager: Polars,
) -> Series[pl.Series]: ...
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
        start, end = DateRange.ensure_py_scalars(start, end, eager)
        return (
            plugins.manager()
            .series(eager, _MAIN)
            .date_range(start, end, days, closed=closed)
            .to_narwhals()
        )
    return (
        DateRange(interval=days, closed=closed)
        .to_function_expr(*_parse.into_iter_expr_ir(start, end))
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
    eager: Polars,
) -> Series[pl.Series]: ...
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
        start, end = LinearSpace.ensure_py_scalars(start, end, eager)
        return (
            plugins.manager()
            .series(eager, _MAIN)
            .linear_space(start, end, num_samples, closed=closed)
            .to_narwhals()
        )
    return (
        LinearSpace(num_samples=num_samples, closed=closed)
        .to_function_expr(*_parse.into_iter_expr_ir(start, end))
        .to_narwhals()
    )


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
