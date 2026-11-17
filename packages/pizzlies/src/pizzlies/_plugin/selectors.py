from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals.compliant import CompliantSelector, CompliantSelectorNamespace
from pizzlies._plugin.expr import DictExpr

if TYPE_CHECKING:
    from collections.abc import Iterator

    from narwhals.dtypes import DType
    from pizzlies._plugin.dataframe import DictDataFrame
    from pizzlies._plugin.series import DictSeries


class DictSelectorNamespace(CompliantSelectorNamespace["DictDataFrame", "DictSeries"]):
    def _iter_schema(self, df: DictDataFrame, /) -> Iterator[tuple[str, DType]]:
        yield from df.schema.items()

    def _iter_columns(self, df: DictDataFrame, /) -> Iterator[DictSeries]:
        yield from df.iter_columns()

    def _iter_columns_dtypes(
        self, df: DictDataFrame, /
    ) -> Iterator[tuple[DictSeries, DType]]:
        for series in self._iter_columns(df):
            yield series, series.dtype

    @property
    def _selector(self) -> type[DictSelector]:
        return DictSelector


class DictSelector(CompliantSelector["DictDataFrame", "DictSeries"], DictExpr):  # type: ignore[misc]
    def _to_expr(self) -> DictExpr:
        return DictExpr(
            self._call,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            version=self._version,
        )
