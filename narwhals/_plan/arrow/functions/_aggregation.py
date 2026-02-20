from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING, Literal

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._plan.arrow import compat, options as pa_options
from narwhals._plan.arrow.functions import _categorical as cat
from narwhals._plan.arrow.functions._arithmetic import power, sub
from narwhals._plan.arrow.functions._construction import array, chunked_array, lit
from narwhals._plan.arrow.functions._dtypes import F64
from narwhals._plan.arrow.functions.meta import call

if TYPE_CHECKING:
    from collections.abc import Callable

    from narwhals._plan.arrow.typing import (
        Arrow,
        ChunkedArray,
        ChunkedArrayAny,
        ChunkedOrArray,
        ChunkedOrArrayAny,
        DataTypeT,
        Scalar,
        ScalarAny,
    )

__all__ = [
    "count",
    "first",
    "implode",
    "kurtosis_skew",
    "last",
    "max",
    "mean",
    "median",
    "min",
    "mode_all",
    "mode_any",
    "n_unique",
    "null_count",
    "quantile",
    "std",
    "sum",
    "var",
]


min = pc.min
"""Get the minimal value in this array."""
max = pc.max
"""Get the maximum value in this array."""
mean = t.cast("Callable[[ChunkedOrArray[pc.NumericScalar]], pa.DoubleScalar]", pc.mean)
"""Reduce this array to the mean value."""
count = pc.count
"""Return the number of non-null elements in this array."""
median = pc.approximate_median
"""Get the median of this array."""
std = pc.stddev
"""Get the standard deviation of this array."""
var = pc.variance
"""Get the variance of this array."""
quantile = pc.quantile
"""Get the quantile value of this array."""


def sum(native: ChunkedOrArrayAny) -> ScalarAny:
    """Reduce this array to the sum value."""
    opts = pa_options.scalar_aggregate(ignore_nulls=True)
    result: ScalarAny = call("sum", native, options=opts)
    return result


def first(native: ChunkedOrArrayAny) -> ScalarAny:
    """Get the first element of this array."""
    return pc.first(native, options=pa_options.scalar_aggregate())


def last(native: ChunkedOrArrayAny) -> ScalarAny:
    """Get the last element of this array."""
    return pc.last(native, options=pa_options.scalar_aggregate())


def implode(native: Arrow[Scalar[DataTypeT]]) -> pa.ListScalar[DataTypeT]:
    """Aggregate values into a list.

    Arguments:
        native: Any arrow data.

    The returned list *itself* is a scalar value of `list` dtype.
    """
    arr = array(native)
    return pa.ListArray.from_arrays([0, len(arr)], arr)[0]


def kurtosis_skew(
    native: ChunkedArray[pc.NumericScalar], function: Literal["kurtosis", "skew"], /
) -> ScalarAny:
    """Compute the kurtosis or sample skewness of this array."""
    result: ScalarAny
    if compat.HAS_KURTOSIS_SKEW:
        if pa.types.is_null(native.type):
            native = native.cast(F64)
        result = call(function, native)
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


def n_unique(native: ChunkedOrArrayAny) -> pa.Int64Scalar:
    """Return the number of unique values in this array."""
    return pc.count_distinct(native, mode="all")


def null_count(native: ChunkedOrArrayAny) -> pa.Int64Scalar:
    """Count the null values in this array."""
    return pc.count(native, mode="only_null")


def mode_any(native: ChunkedOrArrayAny) -> ScalarAny:
    """Compute the most occurring value(s) and return *any* one of them."""
    return first(pc.mode(native, n=1).field("mode"))


def mode_all(native: ChunkedOrArrayAny) -> ChunkedArrayAny:
    """Compute the most occurring value(s) and return *all* of them."""
    struct_arr = pc.mode(native, n=len(native))
    indices = cat.encode(struct_arr.field("count"))
    index_true_modes = lit(0)
    return chunked_array(
        struct_arr.field("mode").filter(pc.equal(indices, index_true_modes))
    )
