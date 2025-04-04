from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._arrow.expr import ArrowExpr
from narwhals._compliant import CompliantSelector
from narwhals._compliant import EagerSelectorNamespace

if TYPE_CHECKING:
    from narwhals._arrow.dataframe import ArrowDataFrame  # noqa: F401
    from narwhals._arrow.series import ArrowSeries  # noqa: F401


class ArrowSelectorNamespace(EagerSelectorNamespace["ArrowDataFrame", "ArrowSeries"]):
    @property
    def _selector(self) -> type[ArrowSelector]:
        return ArrowSelector


class ArrowSelector(CompliantSelector["ArrowDataFrame", "ArrowSeries"], ArrowExpr):  # type: ignore[misc]
    def _to_expr(self) -> ArrowExpr:
        return ArrowExpr(
            self._call,
            depth=self._depth,
            function_name=self._function_name,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )
