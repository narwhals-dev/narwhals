"""Range generation functions."""

from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING, Any, Literal, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._plan.arrow import compat
from narwhals._plan.arrow.functions._arithmetic import add, multiply
from narwhals._plan.arrow.functions._construction import chunked_array, lit
from narwhals._plan.arrow.functions._dtypes import DATE, F64, I32, I64

if TYPE_CHECKING:
    import datetime as dt

    from typing_extensions import TypeAlias

    from narwhals._plan.arrow.typing import (
        Array,
        ArrayAny,
        ChunkedArray,
        ChunkedOrArray,
        DateScalar,
        IntegerScalar,
        IntegerType,
    )
    from narwhals.typing import ClosedInterval

Incomplete: TypeAlias = Any

__all__ = ["date_range", "int_range", "linear_space"]


@overload
def int_range(
    start: int = ...,
    end: int | None = ...,
    step: int = ...,
    /,
    *,
    dtype: IntegerType = ...,
    chunked: Literal[True] = ...,
) -> ChunkedArray[IntegerScalar]: ...
@overload
def int_range(
    start: int = ...,
    end: int | None = ...,
    step: int = ...,
    /,
    *,
    chunked: Literal[False],
) -> pa.Int64Array: ...
@overload
def int_range(
    start: int = ...,
    end: int | None = ...,
    step: int = ...,
    /,
    *,
    dtype: IntegerType = ...,
    chunked: Literal[False],
) -> Array[IntegerScalar]: ...
def int_range(
    start: int = 0,
    end: int | None = None,
    step: int = 1,
    /,
    *,
    dtype: IntegerType = I64,
    chunked: bool = True,
) -> ChunkedOrArray[IntegerScalar]:
    if end is None:
        end = start
        start = 0
    if not compat.HAS_ARANGE:  # pragma: no cover
        import numpy as np  # ignore-banned-import

        arr = pa.array(np.arange(start, end, step), type=dtype)
    else:
        int_range_: Incomplete = pa.arange  # type: ignore[attr-defined]
        arr = t.cast("ArrayAny", int_range_(start, end, step)).cast(dtype)
    return arr if not chunked else pa.chunked_array([arr])


def date_range(
    start: dt.date,
    end: dt.date,
    interval: int,  # (* assuming the `Interval` part is solved)
    *,
    closed: ClosedInterval = "both",
) -> ChunkedArray[DateScalar]:
    start_i = pa.scalar(start).cast(I32).as_py()
    end_i = pa.scalar(end).cast(I32).as_py()
    ca = int_range(start_i, end_i + 1, interval, dtype=I32)
    if closed == "both":
        return ca.cast(DATE)
    if closed == "left":
        ca = ca.slice(length=ca.length() - 1)
    elif closed == "none":
        ca = ca.slice(1, length=ca.length() - 1)
    else:
        ca = ca.slice(1)
    return ca.cast(DATE)


def linear_space(
    start: float, end: float, num_samples: int, *, closed: ClosedInterval = "both"
) -> ChunkedArray[pc.NumericScalar]:
    """Based on [`new_linear_space_f64`].

    [`new_linear_space_f64`]: https://github.com/pola-rs/polars/blob/1684cc09dfaa46656dfecc45ab866d01aa69bc78/crates/polars-ops/src/series/ops/linear_space.rs#L62-L94
    """
    if num_samples < 0:
        msg = f"Number of samples, {num_samples}, must be non-negative."
        raise ValueError(msg)
    if num_samples == 0:
        return chunked_array([[]], F64)
    if num_samples == 1:
        if closed == "none":
            value = (end + start) * 0.5
        elif closed in {"left", "both"}:
            value = float(start)
        else:
            value = float(end)
        return chunked_array([[value]], F64)
    n = num_samples
    span = float(end - start)
    if closed == "none":
        d = span / (n + 1)
        start = start + d
    elif closed == "left":
        d = span / n
    elif closed == "right":
        start = start + span / n
        d = span / n
    else:
        d = span / (n - 1)
    ca = multiply(int_range(0, n).cast(F64), lit(d))
    ca = add(ca, lit(start, F64))
    return ca  # noqa: RET504
