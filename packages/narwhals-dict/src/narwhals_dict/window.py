from __future__ import annotations

from operator import attrgetter
from typing import TYPE_CHECKING, Any

from narwhals_dict.series import DictSeries

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from narwhals._utils import Version

# Leaf node names (see `ExprNode` in `narwhals._expression_parsing`) the kernel
# supports; each is the name of the identically-behaving `DictSeries` method.
# Value-fill `fill_null(value)` is elementwise and never reaches the window path
# (narwhals rejects `over` on elementwise expressions), so only strategy-fill
# arrives here.
GROUPED_WINDOW_LEAVES: frozenset[str] = frozenset(
    {
        "cum_sum",
        "cum_prod",
        "cum_max",
        "cum_min",
        "cum_count",
        "shift",
        "diff",
        "fill_null",
        "is_first_distinct",
        "is_last_distinct",
        "is_unique",
        "rank",
        "rolling_sum",
        "rolling_mean",
        "rolling_var",
        "rolling_std",
        "ewm_mean",
    }
)


def apply_window_kernel(
    method_name: str,
    column: Sequence[Any],
    groups: Mapping[Any, list[int]],
    num_rows: int,
    kwargs: Mapping[str, Any],
    version: Version,
) -> list[Any]:
    """Run `DictSeries.<method_name>(**kwargs)` on each group's ordered column.

    `groups` maps each partition key to its row indices already in `order_by`
    order, so gathering them into a `DictSeries` and calling the leaf method
    reproduces exactly what the general path computes per partition. The result
    is aligned to the original row positions.
    """
    method = attrgetter(method_name)
    out: list[Any] = [None] * num_rows
    for indices in groups.values():
        series = DictSeries([column[i] for i in indices], name="", version=version)
        for i, value in zip(indices, method(series)(**kwargs).native, strict=True):
            out[i] = value
    return out
