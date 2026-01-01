"""Native functions, aliased and/or with behavior aligned to `polars`.

- [ ] _aggregation
  - [ ] Move `list.implode` here too
- [x] _bin_op
- [x] _boolean
- [x] _categorical -> `cat`
- [ ] _common (temp)
- [x] _construction
- [x] _cumulative
- [x] _dtypes
- [ ] _lists
  - [x] -> `list_` (until `functions.__init__` is cleaner)
  - [ ] -> `list`
- [ ] _multiplex
  - [x] move everything
  - [ ] decide on name
- [x] _ranges
- [x] _repeat
- [x] _sort
- [ ] _strings
  - [x] -> `str_` (until `functions.__init__` is cleaner)
  - [ ] -> `str`
- [x] _struct -> `struct`
- [ ] (Others)
  - Vector
  - [Arithmetic](https://arrow.apache.org/docs/python/api/compute.html#arithmetic-functions)
    - some need to be upstream from `_binop`
  - [Rounding](https://arrow.apache.org/docs/python/api/compute.html#rounding-functions)
"""

from __future__ import annotations

import math
import typing as t
from typing import TYPE_CHECKING, Any, Literal

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._plan.arrow import compat, options as pa_options
from narwhals._plan.arrow.functions import (  # noqa: F401
    _categorical as cat,
    _lists as list_,
    _strings as str_,
    _struct as struct,
)
from narwhals._plan.arrow.functions._bin_op import (
    add as add,
    and_ as and_,
    binary as binary,
    eq as eq,
    floordiv as floordiv,
    gt as gt,
    gt_eq as gt_eq,
    lt as lt,
    lt_eq as lt_eq,
    modulus as modulus,
    multiply as multiply,
    not_eq as not_eq,
    or_ as or_,
    power as power,
    sub as sub,
    truediv as truediv,
    xor as xor,
)
from narwhals._plan.arrow.functions._boolean import (
    BOOLEAN_LENGTH_PRESERVING as BOOLEAN_LENGTH_PRESERVING,
    all_ as all_,  # TODO @dangotbanned: Import as `all` when namespace is cleaner
    any_ as any_,  # TODO @dangotbanned: Import as `any` when namespace is cleaner
    eq_missing as eq_missing,
    is_between as is_between,
    is_finite as is_finite,
    is_in as is_in,
    is_nan as is_nan,
    is_not_nan as is_not_nan,
    is_not_null as is_not_null,
    is_null as is_null,
    is_only_nulls as is_only_nulls,
    not_ as not_,
    unique_keep_boolean_length_preserving as unique_keep_boolean_length_preserving,
)
from narwhals._plan.arrow.functions._common import reverse as reverse, round as round
from narwhals._plan.arrow.functions._construction import (
    array as array,
    chunked_array as chunked_array,
    concat_horizontal as concat_horizontal,
    concat_tables as concat_tables,
    concat_vertical as concat_vertical,
    lit as lit,
    to_table as to_table,
)
from narwhals._plan.arrow.functions._cumulative import (
    cum_count as cum_count,
    cum_max as cum_max,
    cum_min as cum_min,
    cum_prod as cum_prod,
    cum_sum as cum_sum,
    cumulative as cumulative,
)
from narwhals._plan.arrow.functions._dtypes import (
    BOOL as BOOL,
    DATE as DATE,
    F64 as F64,
    I32 as I32,
    I64 as I64,
    U32 as U32,
    cast as cast,
    cast_table as cast_table,
    dtype_native as dtype_native,
    string_type as string_type,
)
from narwhals._plan.arrow.functions._lists import ExplodeBuilder as ExplodeBuilder
from narwhals._plan.arrow.functions._multiplex import (
    drop_nulls as drop_nulls,
    fill_nan as fill_nan,
    fill_null as fill_null,
    fill_null_with_strategy as fill_null_with_strategy,
    preserve_nulls as preserve_nulls,
    replace_strict as replace_strict,
    replace_strict_default as replace_strict_default,
    replace_with_mask as replace_with_mask,
    when_then as when_then,
)
from narwhals._plan.arrow.functions._ranges import (
    date_range as date_range,
    int_range as int_range,
    linear_space as linear_space,
)
from narwhals._plan.arrow.functions._repeat import (
    nulls_like as nulls_like,
    repeat as repeat,
    repeat_like as repeat_like,
    repeat_unchecked as repeat_unchecked,
    zeros as zeros,
)
from narwhals._plan.arrow.functions._sort import (
    random_indices as random_indices,
    sort_indices as sort_indices,
    unsort_indices as unsort_indices,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from typing_extensions import TypeAlias

    from narwhals._arrow.typing import Incomplete
    from narwhals._plan.arrow.typing import (
        ChunkedArray,
        ChunkedArrayAny,
        ChunkedOrArray,
        ChunkedOrArrayAny,
        ChunkedOrArrayT,
        ChunkedOrScalarAny,
        NativeScalar,
        NumericScalar,
        ScalarAny,
        UnaryNumeric,
    )
    from narwhals._plan.options import RankOptions
    from narwhals.typing import NonNestedLiteral


abs_ = t.cast("UnaryNumeric", pc.abs)
exp = t.cast("UnaryNumeric", pc.exp)
sqrt = t.cast("UnaryNumeric", pc.sqrt)
ceil = t.cast("UnaryNumeric", pc.ceil)
floor = t.cast("UnaryNumeric", pc.floor)


def sum_(native: Incomplete) -> NativeScalar:
    return pc.sum(native, min_count=0)


def first(native: ChunkedOrArrayAny) -> NativeScalar:
    return pc.first(native, options=pa_options.scalar_aggregate())


def last(native: ChunkedOrArrayAny) -> NativeScalar:
    return pc.last(native, options=pa_options.scalar_aggregate())


min_ = pc.min
# TODO @dangotbanned: Wrap horizontal functions with correct typing
# Should only return scalar if all elements are as well
min_horizontal = pc.min_element_wise
max_ = pc.max
max_horizontal = pc.max_element_wise
mean = t.cast("Callable[[ChunkedOrArray[pc.NumericScalar]], pa.DoubleScalar]", pc.mean)
count = pc.count
median = pc.approximate_median
std = pc.stddev
var = pc.variance
quantile = pc.quantile


def mode_all(native: ChunkedArrayAny) -> ChunkedArrayAny:
    struct = pc.mode(native, n=len(native))
    indices: pa.Int32Array = struct.field("count").dictionary_encode().indices  # type: ignore[attr-defined]
    index_true_modes = lit(0)
    return chunked_array(struct.field("mode").filter(pc.equal(indices, index_true_modes)))


def mode_any(native: ChunkedArrayAny) -> NativeScalar:
    return first(pc.mode(native, n=1).field("mode"))


def kurtosis_skew(
    native: ChunkedArray[pc.NumericScalar], function: Literal["kurtosis", "skew"], /
) -> NativeScalar:
    result: NativeScalar
    if compat.HAS_KURTOSIS_SKEW:
        if pa.types.is_null(native.type):
            native = native.cast(F64)
        result = getattr(pc, function)(native)
    else:
        non_null = native.drop_null()
        if len(non_null) == 0:
            result = lit(None, F64)
        elif len(non_null) == 1:
            result = lit(float("nan"))
        elif function == "skew" and len(non_null) == 2:
            result = lit(0.0, F64)
        else:
            m = sub(non_null, mean(non_null))
            m2 = mean(power(m, lit(2)))
            if function == "kurtosis":
                m4 = mean(power(m, lit(4)))
                result = sub(pc.divide(m4, power(m2, lit(2))), lit(3))
            else:
                m3 = mean(power(m, lit(3)))
                result = pc.divide(m3, power(m2, lit(1.5)))
    return result


def clip_lower(
    native: ChunkedOrScalarAny, lower: ChunkedOrScalarAny
) -> ChunkedOrScalarAny:
    return max_horizontal(native, lower)


def clip_upper(
    native: ChunkedOrScalarAny, upper: ChunkedOrScalarAny
) -> ChunkedOrScalarAny:
    return min_horizontal(native, upper)


def clip(
    native: ChunkedOrScalarAny, lower: ChunkedOrScalarAny, upper: ChunkedOrScalarAny
) -> ChunkedOrScalarAny:
    return clip_lower(clip_upper(native, upper), lower)


def n_unique(native: Any) -> pa.Int64Scalar:
    return count(native, mode="all")


def log(native: ChunkedOrScalarAny, base: float = math.e) -> ChunkedOrScalarAny:
    return t.cast("ChunkedOrScalarAny", pc.logb(native, lit(base)))


def diff(native: ChunkedOrArrayT, n: int = 1) -> ChunkedOrArrayT:
    # pyarrow.lib.ArrowInvalid: Vector kernel cannot execute chunkwise and no chunked exec function was defined
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


def null_count(native: ChunkedOrArrayAny) -> pa.Int64Scalar:
    return pc.count(native, mode="only_null")


SearchSortedSide: TypeAlias = Literal["left", "right"]


# NOTE @dangotbanned: (wish) replacing `np.searchsorted`?
@t.overload
def search_sorted(
    native: ChunkedOrArrayT,
    element: ChunkedOrArray[NumericScalar] | Sequence[float],
    *,
    side: SearchSortedSide = ...,
) -> ChunkedOrArrayT: ...
# NOTE: scalar case may work with only `partition_nth_indices`?
@t.overload
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
