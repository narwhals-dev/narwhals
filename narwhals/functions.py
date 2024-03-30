from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Iterable
from typing import Literal

from narwhals.translate import from_native

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
    if first_item._is_polars:
        import polars as pl

        return from_native(pl.concat([df._dataframe for df in items], how=how))
    from narwhals._pandas_like.namespace import PandasNamespace

    plx = PandasNamespace(first_item._dataframe._implementation)
    return first_item._from_dataframe(
        plx.concat([df._dataframe for df in items], how=how)
    )
