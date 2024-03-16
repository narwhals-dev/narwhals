from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Iterable

if TYPE_CHECKING:
    from narwhals.dataframe import DataFrame
    from narwhals.dataframe import LazyFrame
    from narwhals.typing import IntoExpr

# todo: make groupby and lazygroupby


class GroupBy:
    def __init__(self, df: DataFrame | LazyFrame, *keys: str | Iterable[str]) -> None:
        self._df = df
        self._keys = keys

    def agg(
        self, *aggs: IntoExpr | Iterable[IntoExpr], **named_aggs: IntoExpr
    ) -> DataFrame | LazyFrame:
        aggs, named_aggs = self._df._flatten_and_extract(*aggs, **named_aggs)
        return self._df.__class__(
            self._df._dataframe.group_by(*self._keys).agg(*aggs, **named_aggs),
            implementation=self._df._implementation,
        )
