from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Iterator

from narwhals._dask.expr import DaskExpr
from narwhals._selectors import CompliantSelector
from narwhals._selectors import LazySelectorNamespace

if TYPE_CHECKING:
    try:
        import dask.dataframe.dask_expr as dx
    except ModuleNotFoundError:
        import dask_expr as dx

    from typing_extensions import Self

    from narwhals._dask.dataframe import DaskLazyFrame
    from narwhals._selectors import EvalNames
    from narwhals._selectors import EvalSeries
    from narwhals.utils import _FullContext

    try:
        import dask.dataframe.dask_expr as dx
    except ModuleNotFoundError:
        import dask_expr as dx


class DaskSelectorNamespace(LazySelectorNamespace["DaskLazyFrame", "dx.Series"]):
    def _iter_columns(self, df: DaskLazyFrame) -> Iterator[dx.Series]:
        for _col, ser in df._native_frame.items():  # noqa: PERF102
            yield ser

    def _selector(
        self,
        context: _FullContext,
        call: EvalSeries[DaskLazyFrame, dx.Series],
        evaluate_output_names: EvalNames[DaskLazyFrame],
        /,
    ) -> CompliantSelector[DaskLazyFrame, dx.Series]:
        return DaskSelector(
            call,
            depth=0,
            function_name="selector",
            evaluate_output_names=evaluate_output_names,
            alias_output_names=None,
            backend_version=context._backend_version,
            version=context._version,
        )

    def __init__(self: Self, context: _FullContext, /) -> None:
        self._implementation = context._implementation
        self._backend_version = context._backend_version
        self._version = context._version


class DaskSelector(CompliantSelector["DaskLazyFrame", "dx.Series"], DaskExpr):  # type: ignore[misc]
    @property
    def selectors(self) -> DaskSelectorNamespace:
        return DaskSelectorNamespace(self)

    def __repr__(self: Self) -> str:  # pragma: no cover
        return f"DaskSelector(depth={self._depth}, function_name={self._function_name})"

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
