from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant import CompliantSelector
from narwhals._compliant import EagerSelectorNamespace
from narwhals._pandas_like.dataframe import PandasLikeDataFrame
from narwhals._pandas_like.expr import PandasLikeExpr
from narwhals._pandas_like.series import PandasLikeSeries

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._compliant import EvalNames
    from narwhals._compliant import EvalSeries
    from narwhals._pandas_like.dataframe import PandasLikeDataFrame
    from narwhals._pandas_like.series import PandasLikeSeries


class PandasSelectorNamespace(
    EagerSelectorNamespace["PandasLikeDataFrame", "PandasLikeSeries"]
):
    def _selector(
        self,
        call: EvalSeries[PandasLikeDataFrame, PandasLikeSeries],
        evaluate_output_names: EvalNames[PandasLikeDataFrame],
        /,
    ) -> PandasSelector:
        return PandasSelector(
            call,
            depth=0,
            function_name="selector",
            evaluate_output_names=evaluate_output_names,
            alias_output_names=None,
            implementation=self._implementation,
            backend_version=self._backend_version,
            version=self._version,
        )


class PandasSelector(  # type: ignore[misc]
    CompliantSelector["PandasLikeDataFrame", "PandasLikeSeries"], PandasLikeExpr
):
    def _to_expr(self: Self) -> PandasLikeExpr:
        return PandasLikeExpr(
            self._call,
            depth=self._depth,
            function_name=self._function_name,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            implementation=self._implementation,
            backend_version=self._backend_version,
            version=self._version,
        )
