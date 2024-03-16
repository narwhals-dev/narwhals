from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Generic
from typing import Iterable

if TYPE_CHECKING:
    from narwhals.dataframe import DataFrame
    from narwhals.typing import IntoExpr
from narwhals.typing import T


class GroupBy(Generic[T]):
    def __init__(self, df: DataFrame[T], *keys: str | Iterable[str]) -> None:
        self._df = df
        self._keys = keys

    def agg(
        self, *aggs: IntoExpr | Iterable[IntoExpr], **named_aggs: IntoExpr
    ) -> DataFrame[T]:
        aggs, named_aggs = self._df._flatten_and_extract(*aggs, **named_aggs)
        return self._df.__class__(
            self._df._dataframe.group_by(*self._keys).agg(*aggs, **named_aggs),
            implementation=self._df._implementation,
            features=self._df._features,
        )
