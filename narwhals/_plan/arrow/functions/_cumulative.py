"""https://arrow.apache.org/docs/python/api/compute.html#cumulative-functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow.compute as pc  # ignore-banned-import

from narwhals._plan.arrow.functions._boolean import is_not_null
from narwhals._plan.arrow.functions._common import reverse
from narwhals._plan.arrow.functions._dtypes import U32
from narwhals._plan.expressions import functions as F

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from narwhals._plan.arrow.typing import ChunkedArrayAny, ChunkedOrArrayT


__all__ = ["cum_count", "cum_max", "cum_min", "cum_prod", "cum_sum", "cumulative"]


def cum_sum(native: ChunkedOrArrayT) -> ChunkedOrArrayT:
    """Get an array with the cumulative sum computed at every element."""
    return pc.cumulative_sum(native, skip_nulls=True)


def cum_min(native: ChunkedOrArrayT) -> ChunkedOrArrayT:
    """Get an array with the cumulative min computed at every element."""
    return pc.cumulative_min(native, skip_nulls=True)


def cum_max(native: ChunkedOrArrayT) -> ChunkedOrArrayT:
    """Get an array with the cumulative max computed at every element."""
    return pc.cumulative_max(native, skip_nulls=True)


def cum_prod(native: ChunkedOrArrayT) -> ChunkedOrArrayT:
    """Get an array with the cumulative product computed at every element."""
    return pc.cumulative_prod(native, skip_nulls=True)


def cum_count(native: ChunkedArrayAny) -> ChunkedArrayAny:
    """Return the cumulative count of the non-null values in the array."""
    return cum_sum(is_not_null(native).cast(U32))


def cumulative(native: ChunkedArrayAny, f: F.CumAgg, /) -> ChunkedArrayAny:
    """Dispatch on the cumulative function `f`."""
    func = _CUMULATIVE[type(f)]
    return func(native) if not f.reverse else reverse(func(reverse(native)))


_CUMULATIVE: Mapping[type[F.CumAgg], Callable[[ChunkedArrayAny], ChunkedArrayAny]] = {
    F.CumSum: cum_sum,
    F.CumCount: cum_count,
    F.CumMin: cum_min,
    F.CumMax: cum_max,
    F.CumProd: cum_prod,
}
