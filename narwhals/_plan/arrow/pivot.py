from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from narwhals._plan.arrow import acero, functions as fn, group_by, options as pa_options
from narwhals._plan.arrow.group_by import AggSpec
from narwhals._plan.common import temp
from narwhals._plan.expressions import aggregation as agg

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pyarrow as pa

    from narwhals.typing import PivotAgg


SUPPORTED_PIVOT_AGG: Mapping[PivotAgg, type[agg.AggExpr]] = {
    "min": agg.Min,
    "max": agg.Max,
    "first": agg.First,
    "last": agg.Last,
    "sum": agg.Sum,
    "mean": agg.Mean,
    "median": agg.Median,
    "len": agg.Len,
}


def pivot_table(
    native: pa.Table,
    on: list[str],
    on_columns: pa.Table,
    /,
    index: Sequence[str],
    values: Sequence[str],
    aggregate_function: PivotAgg | None,
    separator: str,
) -> pa.Table:
    """Create a spreadsheet-style `pivot` table.

    Supports  multiple-`on` and aggregations.
    """
    if len(on) == 1:
        on_column = on_columns.column(0)
        on_one = on[0]
        target = native
    else:
        on_column = fn.concat_str(
            '{"', fn.concat_str(*on_columns.columns, separator='","'), '"}'
        )
        on_one = temp.column_name(native.column_names)
        target = acero.join_inner_tables(
            native, on_columns.append_column(on_one, on_column), on
        ).drop(on)
    if aggregate_function:
        target = _aggregate(target, on_one, index, values, aggregate_function)
    return _pivot(target, on_one, on_column.to_pylist(), index, values, separator)


def _replace_flatten_names(
    column_names: list[str],
    /,
    on_columns_names: Sequence[str],
    values: Sequence[str],
    separator: str,
) -> list[str]:
    """Replace the separator used in unnested struct columns.

    [`pa.Table.flatten`] *unconditionally* uses the separator `"."`, so we *likely* need to fix that here.

    [`pa.Table.flatten`]: https://arrow.apache.org/docs/python/generated/pyarrow.Table.html#pyarrow.Table.flatten
    """
    if separator == ".":
        return column_names
    p_on_columns = "|".join(re.escape(name) for name in on_columns_names)
    p_values = "|".join(re.escape(name) for name in values)
    pattern = re.compile(rf"^(?P<on_column>{p_on_columns})\.(?P<value>{p_values})\Z")
    repl = rf"\g<on_column>{separator}\g<value>"
    return [pattern.sub(repl, s) for s in column_names]


def _pivot(
    native: pa.Table,
    on: str,
    on_columns: Sequence[Any],
    /,
    index: Sequence[str],
    values: Sequence[str],
    separator: str,
) -> pa.Table:
    """Perform a single-`on`, non-aggregating `pivot`."""
    options = pa_options.pivot_wider(on_columns)
    specs = (AggSpec((on, name), "hash_pivot_wider", options, name) for name in values)
    pivot = acero.group_by_table(native, index, specs)
    flat = pivot.flatten()
    if len(values) == 1:
        names = [*index, *fn.struct_field_names(pivot.column(values[0]))]
    else:
        names = _replace_flatten_names(flat.column_names, values, on_columns, separator)
    return flat.rename_columns(names)


def _aggregate(
    native: pa.Table,
    on: str,
    /,
    index: Sequence[str],
    values: Sequence[str],
    aggregate_function: PivotAgg,
) -> pa.Table:
    tp_agg = SUPPORTED_PIVOT_AGG[aggregate_function]
    agg_func = group_by.SUPPORTED_AGG[tp_agg]
    option = pa_options.AGG.get(tp_agg)
    specs = (AggSpec(value, agg_func, option) for value in values)
    return acero.group_by_table(native, [*index, on], specs)
