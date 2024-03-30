from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Iterable
from typing import Literal

if TYPE_CHECKING:
    from narwhals.dataframe import DataFrame
    from narwhals.dataframe import LazyFrame


def concat(
    items: Iterable[DataFrame | LazyFrame],
    *,
    how: Literal["horizontal", "vertical"] = "vertical",
) -> DataFrame | LazyFrame:
    if how not in ("horizontal", "vertical"):
        raise NotImplementedError(
            "Only horizontal and vertical concatenations are supported"
        )
    if not items:
        raise ValueError("No items to concatenate")
    items = list(items)
    first_item = items[0]
    plx = first_item.__narwhals_namespace__()
    return first_item.__class__(
        plx.concat([df._dataframe for df in items], how=how),
        is_polars=first_item._is_polars,
    )
