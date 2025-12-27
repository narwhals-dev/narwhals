"""Cached [`pyarrow.compute` options], using `polars` defaults and naming conventions.

See `LazyOptions` for [`__getattr__`] usage.

[`pyarrow.compute` options]: https://arrow.apache.org/docs/dev/python/api/compute.html#compute-options
[`__getattr__`]: https://docs.python.org/3/reference/datamodel.html#module.__getattr__
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc

from narwhals._plan.arrow import compat
from narwhals._utils import zip_strict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import TypeAlias

    from narwhals._plan import expressions as ir
    from narwhals._plan.arrow import acero
    from narwhals._plan.arrow.typing import (
        ChunkedOrArrayAny,
        NullPlacement,
        RankMethodSingle,
    )
    from narwhals._plan.expressions import aggregation as agg
    from narwhals._plan.typing import Order, Seq


__all__ = [
    "AGG",
    "FUNCTION",
    "LIST_AGG",
    "array_sort",
    "count",
    "join",
    "join_replace_nulls",
    "match_substring",
    "rank",
    "scalar_aggregate",
    "sort",
    "split_pattern",
    "variance",
]


_T = TypeVar("_T", bound="type[ir.ExprIR | ir.Function]")

LazyOptions: TypeAlias = Mapping[_T, "acero.AggregateOptions"]
"""Lazily constructed mapping to `pyarrow.compute.FunctionOptions` instances.

Examples:
    >>> from narwhals import _plan as nwp
    >>> from narwhals._plan import expressions as ir
    >>> from narwhals._plan.arrow import options
    >>>
    >>> expr = nwp.col("a").first()
    >>> expr_ir = expr._ir
    >>> expr_ir
    col('a').first()
    >>> if isinstance(expr_ir, ir.AggExpr):
    >>>     print(options.AGG.get(type(expr_ir)))
    ScalarAggregateOptions(skip_nulls=false, min_count=0)

    The first access to `AGG` generated the mapping

    >>> lazy = {"AGG", "FUNCTION", "LIST_AGG"}
    >>> [key for key in options.__dict__ if key in lazy]
    ['AGG']

    We *didn't* generate `FUNCTION`, but it'll be there *when* we need it

    >>> options.FUNCTION.get(ir.functions.NullCount)
    CountOptions(mode=NULLS)

    >>> [key for key in options.__dict__ if key in lazy]
    ['AGG', 'FUNCTION']
"""

AGG: LazyOptions[type[agg.AggExpr]]
FUNCTION: LazyOptions[type[ir.Function]]
LIST_AGG: LazyOptions[type[ir.lists.Aggregation]]


_NULLS_LAST = True
_NULLS_FIRST = False
_ASC = False
_DESC = True

NULL_PLACEMENT: Mapping[bool, NullPlacement] = {
    _NULLS_LAST: "at_end",
    _NULLS_FIRST: "at_start",
}
ORDER: Mapping[bool, Order] = {_ASC: "ascending", _DESC: "descending"}


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


@functools.cache
def array_sort(
    *, descending: bool = False, nulls_last: bool = False
) -> pc.ArraySortOptions:
    return pc.ArraySortOptions(
        order=ORDER[descending], null_placement=NULL_PLACEMENT[nulls_last]
    )


@functools.lru_cache(maxsize=16)
def _sort_key(by: str, *, descending: bool = False) -> tuple[str, Order]:
    return by, ORDER[descending]


@functools.lru_cache(maxsize=8)
def _sort_keys_every(
    by: tuple[str, ...], *, descending: bool = False
) -> Seq[tuple[str, Order]]:
    if len(by) == 1:
        return (_sort_key(by[0], descending=descending),)
    order = ORDER[descending]
    return tuple((key, order) for key in by)


def sort(
    *by: str, descending: bool | Sequence[bool] = False, nulls_last: bool = False
) -> pc.SortOptions:
    if not isinstance(descending, bool) and len(descending) == 1:
        descending = descending[0]
    if isinstance(descending, bool):
        keys = _sort_keys_every(by, descending=descending)
    else:
        it = zip_strict(by, descending)
        keys = tuple(_sort_key(key, descending=desc) for (key, desc) in it)
    return pc.SortOptions(sort_keys=keys, null_placement=NULL_PLACEMENT[nulls_last])


@functools.cache
def rank(
    method: RankMethodSingle, *, descending: bool = False, nulls_last: bool = True
) -> pc.RankOptions:
    return pc.RankOptions(
        sort_keys=ORDER[descending],
        null_placement=NULL_PLACEMENT[nulls_last],
        tiebreaker=("first" if method == "ordinal" else method),
    )


def match_substring(pattern: str) -> pc.MatchSubstringOptions:
    return pc.MatchSubstringOptions(pattern)


def split_pattern(by: str, n: int | None = None) -> pc.SplitPatternOptions:
    """Similar to `str.splitn`.

    Some glue for `max_splits=n - 1`
    """
    if n is not None:
        return pc.SplitPatternOptions(by, max_splits=n - 1)
    return pc.SplitPatternOptions(by)


def pivot_wider(
    on_columns: Sequence[Any] | ChunkedOrArrayAny,
    /,
    unexpected_key_behavior: Literal["ignore", "raise"] = "raise",
) -> pc.FunctionOptions:
    """Tries to wrap [`pc.PivotWiderOptions`], and raises if we're on an old `pyarrow`.

    `key_names` appears to be the same as `on_columns`, but here it is required.

    [`pc.PivotWiderOptions`]: https://arrow.apache.org/docs/python/generated/pyarrow.compute.PivotWiderOptions.html
    """
    if not compat.HAS_PIVOT_WIDER:
        msg = f"`pivot` requires `pyarrow>=20`, got {compat.BACKEND_VERSION!r}"
        raise NotImplementedError(msg)
    opts_cls: Any = pc.PivotWiderOptions  # type: ignore[attr-defined]
    on_cols: Sequence[Any]
    if isinstance(on_columns, (pa.Array, pa.ChunkedArray)):
        on_cols = on_columns.cast(pa.string()).to_pylist()
    else:
        on_cols = on_columns
    options: pc.FunctionOptions = opts_cls(
        on_cols, unexpected_key_behavior=unexpected_key_behavior
    )
    return options


def _generate_agg() -> Mapping[type[agg.AggExpr], acero.AggregateOptions]:
    from narwhals._plan.expressions import aggregation as agg

    return {
        agg.NUnique: count("all"),
        agg.Len: count("all"),
        agg.Count: count("only_valid"),
        agg.First: scalar_aggregate(),
        agg.Last: scalar_aggregate(),
    }


def _generate_list_agg() -> Mapping[type[ir.lists.Aggregation], acero.AggregateOptions]:
    from narwhals._plan.expressions import lists

    return {
        lists.Sum: scalar_aggregate(ignore_nulls=True),
        lists.All: scalar_aggregate(ignore_nulls=True),
        lists.Any: scalar_aggregate(ignore_nulls=True),
        lists.First: scalar_aggregate(),
        lists.Last: scalar_aggregate(),
        lists.NUnique: count("all"),
    }


def _generate_function() -> Mapping[type[ir.Function], acero.AggregateOptions]:
    from narwhals._plan.expressions import boolean, functions

    return {
        boolean.All: scalar_aggregate(ignore_nulls=True),
        boolean.Any: scalar_aggregate(ignore_nulls=True),
        functions.NullCount: count("only_null"),
        functions.Unique: count("all"),
    }


# ruff: noqa: PLW0603
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
        if name == "LIST_AGG":
            global LIST_AGG
            LIST_AGG = _generate_list_agg()
            return LIST_AGG
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
