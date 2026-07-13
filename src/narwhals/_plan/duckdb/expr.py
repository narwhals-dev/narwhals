from __future__ import annotations

from typing import TYPE_CHECKING, Any

from duckdb import Expression as NativeExpr

from narwhals._plan.common import todo
from narwhals._plan.compliant.expr import CompliantExpr
from narwhals._plan.duckdb.classes import DuckDBClasses
from narwhals._utils import Version

if TYPE_CHECKING:
    from typing import TypeAlias

Incomplete: TypeAlias = Any


class DuckDBExpr(CompliantExpr[Incomplete, NativeExpr, NativeExpr]):
    __slots__ = ()
    abs = todo()
    all = todo()
    all_horizontal = todo()
    any = todo()
    any_horizontal = todo()
    arg_max = todo()
    arg_min = todo()
    as_struct = todo()
    binary_expr = todo()
    cast = todo()
    ceil = todo()
    clip = todo()
    clip_lower = todo()
    clip_upper = todo()
    coalesce = todo()
    col = todo()
    concat_str = todo()
    count = todo()
    cum_count = todo()
    cum_max = todo()
    cum_min = todo()
    cum_prod = todo()
    cum_sum = todo()
    date_range = todo()
    diff = todo()
    dispatch = todo()
    drop_nulls = todo()
    ewm_mean = todo()
    exp = todo()
    fill_null = todo()
    fill_null_with_strategy = todo()
    filter = todo()
    first = todo()
    floor = todo()
    hist_bin_count = todo()
    hist_bins = todo()
    int_range = todo()
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
    len_star = todo()
    linear_space = todo()
    lit = todo()
    log = todo()
    max = todo()
    max_horizontal = todo()
    mean = todo()
    mean_horizontal = todo()
    median = todo()
    min = todo()
    min_horizontal = todo()
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
    rolling_mean = todo()
    rolling_std = todo()
    rolling_sum = todo()
    rolling_var = todo()
    round = todo()
    shift = todo()
    skew = todo()
    sort = todo()
    sort_by = todo()
    sqrt = todo()
    std = todo()
    sum = todo()
    sum_horizontal = todo()
    ternary_expr = todo()
    unique = todo()
    var = todo()
    version = Version.MAIN

    @property
    def __narwhals_classes__(self) -> DuckDBClasses:
        return DuckDBClasses()

    cat = todo()  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]
    dt = todo()  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]
    list = todo()  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]
    native = todo()  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]
    str = todo()  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]
    struct = todo()  # pyright: ignore[reportIncompatibleMethodOverride, reportAssignmentType]


DuckDBExpr()
