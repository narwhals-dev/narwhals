from __future__ import annotations

from typing import TYPE_CHECKING

import daft

from narwhals._compliant import CompliantSelector, LazySelectorNamespace
from narwhals._daft.expr import DaftExpr

if TYPE_CHECKING:
    from narwhals._daft.dataframe import DaftLazyFrame  # noqa: F401


class DaftSelectorNamespace(LazySelectorNamespace["DaftLazyFrame", daft.Expression]):
    @property
    def _selector(self) -> type[DaftSelector]:
        return DaftSelector


class DaftSelector(CompliantSelector["DaftLazyFrame", daft.Expression], DaftExpr):  # type: ignore[misc]
    def _to_expr(self) -> DaftExpr:
        return DaftExpr(
            self._call,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )
