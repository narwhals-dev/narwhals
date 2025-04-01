from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant import CompliantSelector
from narwhals._compliant import LazySelectorNamespace
from narwhals._dask.expr import DaskExpr

if TYPE_CHECKING:
    import dask.dataframe.dask_expr as dx  # noqa: F401

    from narwhals._dask.dataframe import DaskLazyFrame  # noqa: F401


class DaskSelectorNamespace(LazySelectorNamespace["DaskLazyFrame", "dx.Series"]):
    @property
    def _selector(self) -> type[DaskSelector]:
        return DaskSelector


class DaskSelector(CompliantSelector["DaskLazyFrame", "dx.Series"], DaskExpr):  # type: ignore[misc]
    def _to_expr(self) -> DaskExpr:
        return DaskExpr(
            self._call,
            depth=self._depth,
            function_name=self._function_name,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )
