"""Cached `pyarrow.compute` options classes, using `polars` defaults.

Important:
    `AGG` and `FUNCTION` mappings are constructed on first `__getattr__` access.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal

import pyarrow.compute as pc  # ignore-banned-import

if TYPE_CHECKING:
    from collections.abc import Mapping

    from narwhals._plan import expressions as ir
    from narwhals._plan.arrow import acero
    from narwhals._plan.expressions import aggregation as agg


__all__ = [
    "AGG",
    "FUNCTION",
    "count",
    "join",
    "join_replace_nulls",
    "scalar_aggregate",
    "variance",
]


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


@functools.cache
def join(*, ignore_nulls: bool = False) -> pc.JoinOptions:
    return pc.JoinOptions(null_handling="skip" if ignore_nulls else "emit_null")


@functools.cache
def join_replace_nulls(*, replacement: str = "__nw_null_value__") -> pc.JoinOptions:
    return pc.JoinOptions(null_handling="replace", null_replacement=replacement)


def _generate_agg() -> Mapping[type[agg.AggExpr], acero.AggregateOptions]:
    from narwhals._plan.expressions import aggregation as agg

    return {
        agg.NUnique: count("all"),
        agg.Len: count("all"),
        agg.Count: count("only_valid"),
        agg.First: scalar_aggregate(),
        agg.Last: scalar_aggregate(),
    }


def _generate_function() -> Mapping[type[ir.Function], acero.AggregateOptions]:
    from narwhals._plan.expressions import boolean, functions

    return {
        boolean.All: scalar_aggregate(ignore_nulls=True),
        boolean.Any: scalar_aggregate(ignore_nulls=True),
        functions.NullCount: count("only_null"),
    }


# ruff: noqa: PLW0603
# NOTE: Using globals for lazy-loading cache
if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:
        if name == "AGG":
            global AGG
            AGG = _generate_agg()
            return AGG
        if name == "FUNCTION":
            global FUNCTION
            FUNCTION = _generate_function()
            return FUNCTION
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
