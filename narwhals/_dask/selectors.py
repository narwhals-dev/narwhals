from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant import CompliantSelector
from narwhals._compliant import LazySelectorNamespace
from narwhals._dask.expr import DaskExpr

if TYPE_CHECKING:
    try:
        import dask.dataframe.dask_expr as dx
    except ModuleNotFoundError:
        import dask_expr as dx

    from typing_extensions import Self

    from narwhals._compliant import EvalNames
    from narwhals._compliant import EvalSeries
    from narwhals._dask.dataframe import DaskLazyFrame


class DaskSelectorNamespace(LazySelectorNamespace["DaskLazyFrame", "dx.Series"]):  # pyright: ignore[reportInvalidTypeArguments]
    def _selector(
        self,
        call: EvalSeries[DaskLazyFrame, dx.Series],  # pyright: ignore[reportInvalidTypeForm]
        evaluate_output_names: EvalNames[DaskLazyFrame],
        /,
    ) -> DaskSelector:
        return DaskSelector(
            call,
            depth=0,
            function_name="selector",
            evaluate_output_names=evaluate_output_names,
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
        )


class DaskSelector(CompliantSelector["DaskLazyFrame", "dx.Series"], DaskExpr):  # type: ignore[misc]
    def _to_expr(self: Self) -> DaskExpr:
        return DaskExpr(
            self._call,
            depth=self._depth,
            function_name=self._function_name,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )
