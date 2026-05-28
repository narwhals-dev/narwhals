"""`pl.Expr` producing functions.

These guys have should be functions/methods that:
- have version compatibility issues
- are used in at least 2 places
"""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Any

import polars as pl

from narwhals._plan.polars import compat

if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = (
    "concat_str",
    "is_finite",
    "is_nan",
    "is_not_nan",
    "len",
    "linear_space",
    "lit",
    "mean_horizontal",
    "over",
    "preserve_nulls",
    "replace_strict",
    "rolling_mean",
    "rolling_std",
    "rolling_sum",
    "rolling_var",
    "row_index",
)

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
            if not partition_by and not compat.OVER_WITHOUT_PARTITION_BY:
                partition_by = (lit(1),)
        return self.over(*partition_by, **options)


if compat.HAS_LINEAR_SPACE or TYPE_CHECKING:
    # NOTE: Has some pretty intricate `@overload`s that can be preserved this way
    linear_space = pl.linear_space
else:

    def linear_space(*_: Any, **__: Any) -> Any:
        raise compat.too_old("linear_space", "1.21.0")


if compat.HAS_LEN or TYPE_CHECKING:
    len = pl.len
else:

    def len() -> pl.Expr:
        return pl.count().alias("len")


def row_index(
    name: str = "index", order_by: Sequence[str] = (), *, nulls_last: bool = False
) -> pl.Expr:
    int_range = pl.int_range(len()).alias(name)
    if not order_by:
        return int_range
    if compat.OVER_RESPECTS_NULLS_LAST:
        return int_range.over(order_by=order_by, nulls_last=nulls_last)
    # NOTE: `nulls_last` isn't the missing feature,
    # but the behavior is more predictable following that change
    by = pl.col(order_by) if builtins.len(order_by) == 1 else pl.struct(order_by)
    return int_range.sort_by(by.arg_sort(nulls_last=nulls_last))


if compat.LIT_ACCEPTS_DICT or TYPE_CHECKING:
    lit = pl.lit
else:

    def lit(value: Any, dtype: pl.DataType | type[pl.DataType] | None = None) -> pl.Expr:
        return pl.struct(**value) if isinstance(value, dict) else pl.lit(value, dtype)


if compat.HAS_MEAN_HORIZONTAL or TYPE_CHECKING:
    mean_horizontal = pl.mean_horizontal

else:

    def mean_horizontal(*exprs: pl.Expr) -> pl.Expr:
        return pl.sum_horizontal(exprs) / pl.sum_horizontal(
            e.is_not_null().cast(pl.Int64) for e in exprs
        )


# TODO @dangotbanned: This seems to good to be true, expect ci to tell me nope
if compat.CONCAT_STR_SUPPORTS_IGNORE_NULLS or TYPE_CHECKING:
    concat_str = pl.concat_str
else:

    def concat_str(
        *exprs: pl.Expr, separator: str = "", ignore_nulls: bool = False
    ) -> pl.Expr:
        if ignore_nulls:
            exprs = (e.fill_null("") for e in exprs)
        return pl.concat_str(*exprs, separator=separator)


def preserve_nulls(before: pl.Expr | pl.Series, after: pl.Expr | pl.Series, /) -> pl.Expr:
    """Propagate nulls positionally from `before` to `after`."""
    return pl.when(before.is_not_null()).then(after)


if compat.IS_NAN_FINITE_NUMERIC_PRESERVES_NULLS or TYPE_CHECKING:
    is_nan = pl.Expr.is_nan
    is_not_nan = pl.Expr.is_not_nan
    is_finite = pl.Expr.is_finite
else:

    def is_nan(self: pl.Expr) -> pl.Expr:
        return preserve_nulls(self, self.is_nan())

    def is_not_nan(self: pl.Expr) -> pl.Expr:
        return preserve_nulls(self, self.is_not_nan())

    def is_finite(self: pl.Expr) -> pl.Expr:
        return preserve_nulls(self, self.is_finite())


if compat.HAS_REPLACE_STRICT or TYPE_CHECKING:
    replace_strict = pl.Expr.replace_strict
else:
    replace_strict = pl.Expr.replace


if compat.ROLLING_VAR_STD_STABLE or TYPE_CHECKING:
    rolling_var = pl.Expr.rolling_var
    rolling_std = pl.Expr.rolling_std

else:

    def rolling_var(*_: Any, **__: Any) -> Any:
        raise compat.too_old("rolling_var", "1.0.0")

    def rolling_std(*_: Any, **__: Any) -> Any:
        raise compat.too_old("rolling_std", "1.0.0")


rolling_sum = pl.Expr.rolling_sum
rolling_mean = pl.Expr.rolling_mean

if compat.EWM_MEAN_PRESERVES_NULLS or TYPE_CHECKING:
    ewm_mean = pl.Expr.ewm_mean
else:

    def ewm_mean(self: pl.Expr, **kwds: Any) -> pl.Expr:
        return preserve_nulls(self, self.ewm_mean(**kwds))
