from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._plan.arrow import compat, options as pa_options
from narwhals._plan.arrow.functions import _struct as struct
from narwhals._plan.arrow.functions._bin_op import not_eq
from narwhals._plan.arrow.functions._boolean import is_between, is_in
from narwhals._plan.arrow.functions._construction import array, chunked_array, lit
from narwhals._plan.arrow.functions._dtypes import BOOL, F64
from narwhals._plan.arrow.functions._multiplex import (
    preserve_nulls,
    replace_with_mask,
    when_then,
)
from narwhals._plan.arrow.functions._ranges import int_range, linear_space
from narwhals._plan.arrow.functions._repeat import repeat_like, zeros

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from narwhals._plan.arrow.typing import (
        ChunkedArray,
        ChunkedArrayAny,
        ChunkedOrArray,
        ChunkedOrArrayT,
        NumericScalar,
        ScalarAny,
        SearchSortedSide,
    )
    from narwhals._plan.options import RankOptions
    from narwhals.typing import NonNestedLiteral


def diff(native: ChunkedOrArrayT, n: int = 1) -> ChunkedOrArrayT:
    return (
        pc.pairwise_diff(native, n)
        if isinstance(native, pa.Array)
        else chunked_array(pc.pairwise_diff(native.combine_chunks(), n))
    )


def shift(
    native: ChunkedArrayAny, n: int, *, fill_value: NonNestedLiteral = None
) -> ChunkedArrayAny:
    if n == 0:
        return native
    arr = native
    if n > 0:
        filled = repeat_like(fill_value, n, arr)
        arrays = [filled, *arr.slice(length=arr.length() - n).chunks]
    else:
        filled = repeat_like(fill_value, -n, arr)
        arrays = [*arr.slice(offset=-n).chunks, filled]
    return pa.chunked_array(arrays)


def rank(native: ChunkedArrayAny, rank_options: RankOptions) -> ChunkedArrayAny:
    arr = native if compat.RANK_ACCEPTS_CHUNKED else array(native)
    if rank_options.method == "average":
        # Adapted from https://github.com/pandas-dev/pandas/blob/f4851e500a43125d505db64e548af0355227714b/pandas/core/arrays/arrow/array.py#L2290-L2316
        order = pa_options.ORDER[rank_options.descending]
        min = preserve_nulls(arr, pc.rank(arr, order, tiebreaker="min").cast(F64))
        max = pc.rank(arr, order, tiebreaker="max").cast(F64)
        ranked = pc.divide(pc.add(min, max), lit(2, F64))
    else:
        ranked = preserve_nulls(native, pc.rank(arr, options=rank_options.to_arrow()))
    return chunked_array(ranked)


# NOTE @dangotbanned: (wish) replacing `np.searchsorted`?
@overload
def search_sorted(
    native: ChunkedOrArrayT,
    element: ChunkedOrArray[NumericScalar] | Sequence[float],
    *,
    side: SearchSortedSide = ...,
) -> ChunkedOrArrayT: ...
# NOTE: scalar case may work with only `partition_nth_indices`?
@overload
def search_sorted(
    native: ChunkedOrArrayT, element: float, *, side: SearchSortedSide = ...
) -> ScalarAny: ...
def search_sorted(
    native: ChunkedOrArrayT,
    element: ChunkedOrArray[NumericScalar] | Sequence[float] | float,
    *,
    side: SearchSortedSide = "left",
) -> ChunkedOrArrayT | ScalarAny:
    """Find indices where elements should be inserted to maintain order."""
    import numpy as np  # ignore-banned-import

    indices = np.searchsorted(element, native, side=side)
    if isinstance(indices, np.generic):
        return lit(indices)
    if isinstance(native, pa.ChunkedArray):
        return chunked_array([indices])
    return array(indices)


def hist_bins(
    native: ChunkedArrayAny,
    bins: Sequence[float] | ChunkedArray[NumericScalar],
    *,
    include_breakpoint: bool,
) -> Mapping[str, Iterable[Any]]:
    """Bin values into buckets and count their occurrences.

    Notes:
        Assumes that the following edge cases have been handled:
        - `len(bins) >= 2`
        - `bins` increase monotonically
        - `bin[0] != bin[-1]`
        - `native` contains values that are non-null (including NaN)
    """
    if len(bins) == 2:
        upper = bins[1]
        count = array(is_between(native, bins[0], upper, closed="both"), BOOL).true_count
        if include_breakpoint:
            return {"breakpoint": [upper], "count": [count]}
        return {"count": [count]}

    # lowest bin is inclusive
    # NOTE: `np.unique` behavior sorts first
    value_counts = (
        when_then(not_eq(native, lit(bins[0])), search_sorted(native, bins), 1)
        .sort()
        .value_counts()
    )
    values, counts = struct.fields(value_counts, "values", "counts")
    bin_count = len(bins)
    int_range_ = int_range(1, bin_count, chunked=False)
    mask = is_in(int_range_, values)
    replacements = counts.filter(is_in(values, int_range_))
    counts = replace_with_mask(zeros(bin_count - 1), mask, replacements)

    if include_breakpoint:
        return {"breakpoint": bins[1:], "count": counts}
    return {"count": counts}


def hist_zeroed_data(
    arg: int | Sequence[float], *, include_breakpoint: bool
) -> Mapping[str, Iterable[Any]]:
    # NOTE: If adding `linear_space` and `zeros` to `CompliantNamespace`, consider moving this.
    n = arg if isinstance(arg, int) else len(arg) - 1
    if not include_breakpoint:
        return {"count": zeros(n)}
    bp = linear_space(0, 1, arg, closed="right") if isinstance(arg, int) else arg[1:]
    return {"breakpoint": bp, "count": zeros(n)}
