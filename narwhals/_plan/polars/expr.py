from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from narwhals._plan.common import todo
from narwhals._plan.compliant.expr import CompliantExpr
from narwhals._plan.polars import compat
from narwhals._plan.polars.classes import PolarsClasses
from narwhals._plan.polars.namespace import dtype_to_native, dtype_to_native_fast
from narwhals._utils import Version

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypeAlias

    from typing_extensions import Self

    from narwhals._plan import expressions as ir
    from narwhals._plan.expressions.ranges import IntRange
    from narwhals._plan.polars.dataframe import PolarsDataFrame as DataFrame
    from narwhals._plan.polars.lazyframe import PolarsLazyFrame as LazyFrame
    from narwhals.typing import IntoDType, PythonLiteral


Incomplete: TypeAlias = Any

if compat.OVER_RESPECTS_NULLS_LAST:
    # NOTE: Allows all features, so no need to branch in any calls
    def over(
        self: pl.Expr,
        *partition_by: pl.Expr | str,
        order_by: Sequence[str] | None = None,
        descending: bool = False,
        nulls_last: bool = False,
    ) -> pl.Expr:
        return self.over(
            *partition_by, order_by=order_by, descending=descending, nulls_last=nulls_last
        )
else:

    def over(
        self: pl.Expr,
        *partition_by: pl.Expr | str,
        order_by: Sequence[str] | None = None,
        descending: bool = False,
        nulls_last: bool = False,
    ) -> pl.Expr:
        if nulls_last:
            raise compat.over_error("nulls_last")
        options: dict[str, Any] = {}
        if order_by:
            if not compat.OVER_SUPPORTS_ORDER_BY:
                raise compat.over_error("order_by_any")
            options["order_by"] = order_by
            if descending:
                if not compat.OVER_SUPPORTS_DESCENDING:
                    raise compat.over_error("descending")
                options["descending"] = descending
            if not partition_by and compat.OVER_WITHOUT_PARTITION_BY:
                raise compat.over_error("order_by_only")
        return self.over(*partition_by, **options)


if compat.HAS_LINEAR_SPACE or TYPE_CHECKING:
    # NOTE: Has some pretty intricate `@overload`s that can be preserved this way
    linear_space = pl.linear_space
else:

    def linear_space(*_: Any, **__: Any) -> Any:
        raise compat.too_old("linear_space", "1.21.0")


# TODO @dangotbanned: (low-priority) Support without triggering
# `over(..., order_by=...)` requires `polars>=1.0.0`
def row_index(
    name: str = "index", order_by: Sequence[str] = (), *, nulls_last: bool = False
) -> pl.Expr:
    expr = pl.int_range(pl.len())
    if order_by:
        expr = over(expr, order_by=order_by, nulls_last=nulls_last)
    return expr.alias(name)


class PolarsExpr(CompliantExpr["DataFrame", pl.Expr, pl.Expr]):
    __slots__ = ("_native",)
    _native: pl.Expr
    version: ClassVar = Version.MAIN

    def _with_native(self, native: pl.Expr, name: str = "", /) -> Self:
        return self.from_native(native, name)

    # NOTE: Unsure how much of `name` might be needed for polars
    @classmethod
    def from_native(cls, native: pl.Expr, name: str = "", /) -> Self:
        obj = cls.__new__(cls)
        obj._native = native if not name else native.alias(name)
        return obj

    @classmethod
    def from_python(
        cls,
        value: PythonLiteral,
        name: str = "literal",
        /,
        *,
        dtype: IntoDType | None = None,
    ) -> Self:
        unknown = cls.version.dtypes.Unknown
        dtype_pl = None if dtype == unknown else dtype_to_native(dtype, cls.version)
        return cls.from_native(pl.lit(value, dtype_pl), name)

    @property
    def native(self) -> pl.Expr:
        return self._native

    @property
    def __narwhals_classes__(self) -> PolarsClasses:
        return PolarsClasses()

    def dispatch(
        self, node: ir.ExprIR, frame: DataFrame | LazyFrame, name: str
    ) -> PolarsExpr:
        """Trying to limit the API surface for now.

        - polars only uses `PolarsDataFrame._evaluate_irs`
        - pyarrow is more tangled up
        """
        return node.__expr_ir_dispatch__(node, self, frame, name)

    @classmethod
    def col(cls, node: ir.Column, _: Incomplete, name: str, /) -> Self:
        return cls.from_native(pl.col(node.name), name)

    @classmethod
    def lit(cls, node: ir.Lit[PythonLiteral], _: Incomplete, name: str, /) -> Self:
        return cls.from_python(node.value, name, dtype=node.dtype)

    @classmethod
    def lit_series(
        cls, node: ir.LitSeries[pl.Series], _: Incomplete, name: str, /
    ) -> Self:
        return cls.from_native(pl.lit(node.native), name)

    @classmethod
    def len_star(cls, _: ir.Len, __: Incomplete, name: str, /) -> Self:
        return cls.from_native(pl.len(), name)

    abs = todo()
    all = todo()
    all_horizontal = todo()
    any = todo()
    any_horizontal = todo()
    arg_max = todo()
    arg_min = todo()
    binary_expr = todo()

    def cast(self, node: ir.Cast, frame: Incomplete, name: str) -> Self:
        dtype = dtype_to_native(node.dtype, self.version)
        return self._with_native(self.dispatch(node.expr, frame, name).native.cast(dtype))

    coalesce = todo()
    ceil = todo()
    clip = todo()
    clip_lower = todo()
    clip_upper = todo()
    concat_str = todo()
    count = todo()
    cum_count = todo()
    cum_max = todo()
    cum_min = todo()
    cum_prod = todo()
    cum_sum = todo()
    date_range = todo()
    diff = todo()
    drop_nulls = todo()
    ewm_mean = todo()
    exp = todo()
    fill_nan = todo()
    fill_null = todo()
    fill_null_with_strategy = todo()
    filter = todo()

    def first(self, node: ir.aggregation.First, frame: Any, name: str) -> Self:
        return self._with_native(self.dispatch(node.expr, frame, name).native.first())

    floor = todo()
    hist_bin_count = todo()
    hist_bins = todo()

    def int_range(
        self, node: ir.RangeExpr[IntRange], frame: Incomplete, name: str
    ) -> Self:
        func = node.function
        if fastpath := func.try_unwrap_literals(node):
            dtype = dtype_to_native_fast(func.dtype)
            native = pl.int_range(*fastpath, func.step, dtype=dtype)
            return self.from_native(native, name)
        msg = f"TODO @dangotbanned: `{self.int_range.__qualname__}()` w/ non-`Lit` inputs, got \n{node.input[0]!r}\n{node.input[1]!r}"
        raise NotImplementedError(msg)

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
    linear_space = todo()
    len = todo()
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
    rolling_sum = todo()
    rolling_mean = todo()
    rolling_std = todo()
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

    cat = todo()  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]
    dt = todo()  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]
    list = todo()  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]
    str = todo()  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]
    struct = todo()  # pyright: ignore[reportAssignmentType, reportIncompatibleMethodOverride]


PolarsExpr()
