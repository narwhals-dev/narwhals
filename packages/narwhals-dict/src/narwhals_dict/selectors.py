from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant import CompliantSelector, EagerSelectorNamespace
from narwhals_dict.expr import DictExpr

if TYPE_CHECKING:
    from narwhals_dict.dataframe import DictDataFrame  # noqa: F401
    from narwhals_dict.series import DictSeries  # noqa: F401


class DictSelectorNamespace(EagerSelectorNamespace["DictDataFrame", "DictSeries"]):
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
