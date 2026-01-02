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
- [ ] _lists
  - [x] -> `list_` (until `functions.__init__` is cleaner)
  - [ ] -> `list`
- [ ] _multiplex
  - [x] move everything
  - [ ] decide on name
- [x] _ranges
- [x] _repeat
- [x] _round
- [x] _sort
- [ ] _strings
  - [x] -> `str_` (until `functions.__init__` is cleaner)
  - [ ] -> `str`
- [x] _struct -> `struct`
- [ ] _vector
- [ ] (Others)
"""

from __future__ import annotations

import math
import typing as t
from typing import TYPE_CHECKING

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._plan.arrow import compat as compat
from narwhals._plan.arrow.functions import (  # noqa: F401
    _categorical as cat,
    _lists as list_,
    _strings as str_,
    _struct as struct,
)
from narwhals._plan.arrow.functions._aggregation import (
    count as count,
    first as first,
    kurtosis_skew as kurtosis_skew,
    last as last,
    max as max,
    mean as mean,
    median as median,
    min as min,
    mode_any as mode_any,
    n_unique as n_unique,
    null_count as null_count,
    quantile as quantile,
    std as std,
    sum as sum,
    var as var,
)
from narwhals._plan.arrow.functions._arithmetic import (
    add as add,
    floordiv as floordiv,
    modulus as modulus,
    multiply as multiply,
    power as power,
    sqrt as sqrt,
    sub as sub,
    truediv as truediv,
)
from narwhals._plan.arrow.functions._bin_op import (
    and_ as and_,
    binary as binary,
    eq as eq,
    gt as gt,
    gt_eq as gt_eq,
    lt as lt,
    lt_eq as lt_eq,
    not_eq as not_eq,
    or_ as or_,
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
from narwhals._plan.arrow.functions._common import reverse as reverse
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
from narwhals._plan.arrow.functions._horizontal import (
    max_horizontal as max_horizontal,
    min_horizontal as min_horizontal,
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
from narwhals._plan.arrow.functions._round import (
    ceil as ceil,
    clip as clip,
    clip_lower as clip_lower,
    clip_upper as clip_upper,
    floor as floor,
    round as round,
)
from narwhals._plan.arrow.functions._sort import (
    random_indices as random_indices,
    sort_indices as sort_indices,
    unsort_indices as unsort_indices,
)
from narwhals._plan.arrow.functions._vector import (
    diff as diff,
    hist_bins as hist_bins,
    hist_zeroed_data as hist_zeroed_data,
    rank as rank,
    search_sorted as search_sorted,
    shift as shift,
)

if TYPE_CHECKING:
    from narwhals._plan.arrow.typing import (
        ChunkedArrayAny,
        ChunkedOrScalarAny,
        UnaryNumeric,
    )


abs_ = t.cast("UnaryNumeric", pc.abs)
exp = t.cast("UnaryNumeric", pc.exp)


def mode_all(native: ChunkedArrayAny) -> ChunkedArrayAny:
    struct_arr = pc.mode(native, n=len(native))
    indices: pa.Int32Array = struct_arr.field("count").dictionary_encode().indices  # type: ignore[attr-defined]
    index_true_modes = lit(0)
    return chunked_array(struct_arr.field("mode").filter(eq(indices, index_true_modes)))


def log(native: ChunkedOrScalarAny, base: float = math.e) -> ChunkedOrScalarAny:
    return t.cast("ChunkedOrScalarAny", pc.logb(native, lit(base)))
