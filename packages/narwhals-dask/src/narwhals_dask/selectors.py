from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals.compliant import CompliantSelector, CompliantSelectorNamespace
from narwhals_dask.expr import DaskExpr

if TYPE_CHECKING:
    from collections.abc import Iterator

    import dask.dataframe.dask_expr as dx

    from narwhals.dtypes import DType
    from narwhals_dask.dataframe import DaskLazyFrame


class DaskSelectorNamespace(CompliantSelectorNamespace["DaskLazyFrame", "dx.Series"]):  # pyright: ignore[reportInvalidTypeArguments]
    @property
    def _selector(self) -> type[DaskSelector]:
        return DaskSelector

    def _iter_schema(self, df: DaskLazyFrame) -> Iterator[tuple[str, DType]]:
        yield from df.schema.items()

    def _iter_columns(self, df: DaskLazyFrame) -> Iterator[dx.Series]:
        yield from df._iter_columns()

    def _iter_columns_dtypes(
        self, df: DaskLazyFrame, /
    ) -> Iterator[tuple[dx.Series, DType]]:
        yield from zip(self._iter_columns(df), df.schema.values(), strict=True)


class DaskSelector(CompliantSelector["DaskLazyFrame", "dx.Series"], DaskExpr):  # pyright: ignore[reportInvalidTypeArguments]
    def _to_expr(self) -> DaskExpr:
        return DaskExpr(
            self._call,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            version=self._version,
        )
