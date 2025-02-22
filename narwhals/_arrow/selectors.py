from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Iterator

from narwhals._arrow.expr import ArrowExpr
from narwhals._selectors import CompliantSelector
from narwhals._selectors import CompliantSelectorNamespace

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.dataframe import ArrowDataFrame
    from narwhals._arrow.series import ArrowSeries
    from narwhals._selectors import EvalNames
    from narwhals._selectors import EvalSeries
    from narwhals.utils import _FullContext


class ArrowSelectorNamespace(CompliantSelectorNamespace["ArrowDataFrame", "ArrowSeries"]):
    def _iter_columns(self, df: ArrowDataFrame) -> Iterator[ArrowSeries]:
        from narwhals._arrow.series import ArrowSeries

        for col, ser in zip(df.columns, df._native_frame.itercolumns()):
            yield ArrowSeries(
                ser, name=col, backend_version=df._backend_version, version=df._version
            )

    def _selector(
        self,
        context: _FullContext,
        call: EvalSeries[ArrowDataFrame, ArrowSeries],
        evaluate_output_names: EvalNames[ArrowDataFrame],
        /,
    ) -> ArrowSelector:
        return ArrowSelector(
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


class ArrowSelector(CompliantSelector["ArrowDataFrame", "ArrowSeries"], ArrowExpr):  # type: ignore[misc]
    @property
    def selectors(self) -> ArrowSelectorNamespace:
        return ArrowSelectorNamespace(self)

    def _to_expr(self: Self) -> ArrowExpr:
        return ArrowExpr(
            self._call,
            depth=self._depth,
            function_name=self._function_name,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
        )
