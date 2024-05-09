from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Iterator

from narwhals.utils import flatten
from narwhals.utils import tupleify

if TYPE_CHECKING:
    from narwhals.dataframe import DataFrame
    from narwhals.dataframe import LazyFrame
    from narwhals.typing import IntoExpr


class GroupBy:
    def __init__(self, df: DataFrame, *keys: str | Iterable[str]) -> None:
        self._df = df
        self._keys = flatten(keys)
        self._grouped = self._df._dataframe.group_by(self._keys)

    def agg(
        self, *aggs: IntoExpr | Iterable[IntoExpr], **named_aggs: IntoExpr
    ) -> DataFrame:
        aggs, named_aggs = self._df._flatten_and_extract(*aggs, **named_aggs)
        return self._df.__class__(
            self._grouped.agg(*aggs, **named_aggs),
        )

    def __iter__(self) -> Iterator[tuple[Any, DataFrame]]:
        import narwhals as nw

        yield from (
            (tupleify(key), nw.from_native(df, eager_only=True))
            for (key, df) in self._grouped.__iter__()
        )


class LazyGroupBy:
    def __init__(self, df: LazyFrame, *keys: str | Iterable[str]) -> None:
        self._df = df
        self._keys = keys
        self._grouped = self._df._dataframe.group_by(*self._keys)

    def agg(
        self, *aggs: IntoExpr | Iterable[IntoExpr], **named_aggs: IntoExpr
    ) -> LazyFrame:
        aggs, named_aggs = self._df._flatten_and_extract(*aggs, **named_aggs)
        return self._df.__class__(
            self._grouped.agg(*aggs, **named_aggs),
        )
