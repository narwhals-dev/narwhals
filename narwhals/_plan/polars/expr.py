from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._plan.common import todo
from narwhals._plan.compliant.expr import CompliantExpr

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

Incomplete: TypeAlias = Any


class PolarsExpr(CompliantExpr[Incomplete, Incomplete]):
    _evaluated = todo()
    _version = todo()  # type: ignore[assignment]
    _with_native = todo()
    abs = todo()
    all = todo()
    any = todo()
    arg_max = todo()
    arg_min = todo()
    binary_expr = todo()
    cast = todo()
    ceil = todo()
    clip = todo()
    clip_lower = todo()
    clip_upper = todo()
    count = todo()
    cum_count = todo()
    cum_max = todo()
    cum_min = todo()
    cum_prod = todo()
    cum_sum = todo()
    diff = todo()
    drop_nulls = todo()
    ewm_mean = todo()
    exp = todo()
    fill_nan = todo()
    fill_null = todo()
    fill_null_with_strategy = todo()
    filter = todo()
    first = todo()
    floor = todo()
    from_native = todo()
    hist_bin_count = todo()
    hist_bins = todo()
    is_between = todo()
    is_duplicated = todo()
    is_finite = todo()
    is_first_distinct = todo()
    is_in_expr = todo()
    is_in_seq = todo()
    is_last_distinct = todo()
    is_nan = todo()
    is_not_nan = todo()
    is_not_null = todo()
    is_null = todo()
    is_unique = todo()
    kurtosis = todo()
    last = todo()
    len = todo()
    log = todo()
    max = todo()
    mean = todo()
    median = todo()
    min = todo()
    mode_all = todo()
    mode_any = todo()
    n_unique = todo()
    not_ = todo()
    null_count = todo()
    over = todo()
    over_ordered = todo()
    pow = todo()
    quantile = todo()
    rank = todo()
    replace_strict = todo()
    replace_strict_default = todo()
    rolling_expr = todo()
    round = todo()
    shift = todo()
    skew = todo()
    sort = todo()
    sort_by = todo()
    sqrt = todo()
    std = todo()
    sum = todo()
    ternary_expr = todo()
    unique = todo()
    var = todo()

    cat = todo()  # type: ignore[assignment]
    list = todo()  # type: ignore[assignment]
    name = todo()  # type: ignore[assignment]
    str = todo()  # type: ignore[assignment]
    struct = todo()  # type: ignore[assignment]
    version = todo()  # type: ignore[assignment]


PolarsExpr()
