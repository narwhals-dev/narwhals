"""Native functions, aliased and/or with behavior aligned to `polars`.

- [x] _aggregation
- [x] _arithmetic
- [x] _bin_op
- [x] _boolean
- [x] _categorical -> `cat`
- [ ] _common (temp)
- [x] _construction
- [x] _cumulative
- [x] _dtypes
- [x] _horizontal
- [x] _lists -> `list`
- [ ] _multiplex
  - [x] move everything
  - [ ] decide on name
- [x] _ranges
- [x] _repeat
- [x] _round
- [x] _sort
- [x] _strings -> `str`
- [x] _struct -> `struct`
- [x] _vector
- [ ] (Others?)
"""

from __future__ import annotations

from narwhals._plan.arrow.functions import (
    _categorical as cat,
    _lists as list,
    _strings as str,
    _struct as struct,
)
from narwhals._plan.arrow.functions._aggregation import (
    count,
    first,
    kurtosis_skew,
    last,
    max,
    mean,
    median,
    min,
    mode_any,
    n_unique,
    null_count,
    quantile,
    std,
    sum,
    var,
)
from narwhals._plan.arrow.functions._arithmetic import (
    add,
    floordiv,
    modulus,
    multiply,
    power,
    sqrt,
    sub,
    truediv,
)
from narwhals._plan.arrow.functions._bin_op import (
    and_,
    binary,
    eq,
    gt,
    gt_eq,
    lt,
    lt_eq,
    not_eq,
    or_,
    xor,
)
from narwhals._plan.arrow.functions._boolean import (
    BOOLEAN_LENGTH_PRESERVING,
    all,
    any,
    eq_missing,
    is_between,
    is_finite,
    is_in,
    is_nan,
    is_not_nan,
    is_not_null,
    is_null,
    is_only_nulls,
    not_,
    unique_keep_boolean_length_preserving,
)
from narwhals._plan.arrow.functions._common import (
    MinMax,
    abs,
    exp,
    is_arrow,
    log,
    mode_all,
)
from narwhals._plan.arrow.functions._construction import (
    array,
    chunked_array,
    concat_horizontal,
    concat_tables,
    concat_vertical,
    lit,
    to_table,
)
from narwhals._plan.arrow.functions._cumulative import (
    cum_count,
    cum_max,
    cum_min,
    cum_prod,
    cum_sum,
    cumulative,
)
from narwhals._plan.arrow.functions._dtypes import (
    BOOL,
    DATE,
    F64,
    I32,
    I64,
    U32,
    cast,
    cast_table,
    dtype_native,
    string_type,
)
from narwhals._plan.arrow.functions._horizontal import max_horizontal, min_horizontal
from narwhals._plan.arrow.functions._lists import ExplodeBuilder
from narwhals._plan.arrow.functions._multiplex import (
    drop_nulls,
    fill_nan,
    fill_null,
    fill_null_with_strategy,
    preserve_nulls,
    replace_strict,
    replace_strict_default,
    replace_with_mask,
    when_then,
)
from narwhals._plan.arrow.functions._ranges import date_range, int_range, linear_space
from narwhals._plan.arrow.functions._repeat import (
    nulls_like,
    repeat,
    repeat_like,
    repeat_unchecked,
    zeros,
)
from narwhals._plan.arrow.functions._round import (
    ceil,
    clip,
    clip_lower,
    clip_upper,
    floor,
    round,
)
from narwhals._plan.arrow.functions._sort import (
    random_indices,
    reverse,
    sort_indices,
    unsort_indices,
)
from narwhals._plan.arrow.functions._vector import (
    diff,
    hist_bins,
    hist_zeroed_data,
    rank,
    search_sorted,
    shift,
)

__all__ = [
    "BOOL",
    "BOOLEAN_LENGTH_PRESERVING",
    "DATE",
    "F64",
    "I32",
    "I64",
    "U32",
    "ExplodeBuilder",
    "MinMax",
    "abs",
    "add",
    "all",
    "and_",
    "any",
    "array",
    "binary",
    "cast",
    "cast_table",
    "cat",
    "ceil",
    "chunked_array",
    "clip",
    "clip_lower",
    "clip_upper",
    "concat_horizontal",
    "concat_tables",
    "concat_vertical",
    "count",
    "cum_count",
    "cum_max",
    "cum_min",
    "cum_prod",
    "cum_sum",
    "cumulative",
    "date_range",
    "diff",
    "drop_nulls",
    "dtype_native",
    "eq",
    "eq_missing",
    "exp",
    "fill_nan",
    "fill_null",
    "fill_null_with_strategy",
    "first",
    "floor",
    "floordiv",
    "gt",
    "gt_eq",
    "hist_bins",
    "hist_zeroed_data",
    "int_range",
    "is_arrow",
    "is_between",
    "is_finite",
    "is_in",
    "is_nan",
    "is_not_nan",
    "is_not_null",
    "is_null",
    "is_only_nulls",
    "kurtosis_skew",
    "last",
    "linear_space",
    "list",
    "lit",
    "log",
    "lt",
    "lt_eq",
    "max",
    "max_horizontal",
    "mean",
    "median",
    "min",
    "min_horizontal",
    "mode_all",
    "mode_any",
    "modulus",
    "multiply",
    "n_unique",
    "not_",
    "not_eq",
    "null_count",
    "nulls_like",
    "or_",
    "power",
    "preserve_nulls",
    "quantile",
    "random_indices",
    "rank",
    "repeat",
    "repeat_like",
    "repeat_unchecked",
    "replace_strict",
    "replace_strict_default",
    "replace_with_mask",
    "reverse",
    "round",
    "search_sorted",
    "shift",
    "sort_indices",
    "sqrt",
    "std",
    "str",
    "string_type",
    "struct",
    "sub",
    "sum",
    "to_table",
    "truediv",
    "unique_keep_boolean_length_preserving",
    "unsort_indices",
    "var",
    "when_then",
    "xor",
    "zeros",
]
