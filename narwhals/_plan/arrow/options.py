"""Cached `pyarrow.compute` options classes, using `polars` defaults."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal

import pyarrow.compute as pc  # ignore-banned-import

if TYPE_CHECKING:
    from collections.abc import Mapping

    from narwhals._plan import expressions as ir
    from narwhals._plan.arrow import acero
    from narwhals._plan.expressions import aggregation as agg


__all__ = ["AGG", "FUNCTION", "count", "scalar_aggregate", "variance"]


AGG: Mapping[type[agg.AggExpr], acero.AggregateOptions]
FUNCTION: Mapping[type[ir.Function], acero.AggregateOptions]


@functools.cache
def count(
    mode: Literal["only_valid", "only_null", "all"] = "only_valid",
) -> pc.CountOptions:
    return pc.CountOptions(mode)


# pyarrow defaults to ignore_nulls
# polars doesn't mention
@functools.cache
def variance(
    ddof: int = 1, *, ignore_nulls: bool = True, min_count: int = 0
) -> pc.VarianceOptions:
    return pc.VarianceOptions(ddof=ddof, skip_nulls=ignore_nulls, min_count=min_count)


@functools.cache
def scalar_aggregate(
    *, ignore_nulls: bool = False, min_count: int = 0
) -> pc.ScalarAggregateOptions:
    return pc.ScalarAggregateOptions(skip_nulls=ignore_nulls, min_count=min_count)


# ruff: noqa: PLW0603
# NOTE: Using globals for lazy-loading cache
if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:
        if name == "AGG":
            from narwhals._plan.expressions import aggregation as agg

            global AGG
            AGG = {
                agg.NUnique: count("all"),
                agg.Len: count("all"),
                agg.Count: count("only_valid"),
                agg.First: scalar_aggregate(),
                agg.Last: scalar_aggregate(),
            }
            return AGG
        if name == "FUNCTION":
            from narwhals._plan.expressions import boolean

            global FUNCTION
            FUNCTION = {
                boolean.All: scalar_aggregate(ignore_nulls=True),
                boolean.Any: scalar_aggregate(ignore_nulls=True),
            }
            return FUNCTION
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
