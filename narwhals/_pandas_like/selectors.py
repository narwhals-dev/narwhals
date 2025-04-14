from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant import CompliantSelector
from narwhals._compliant import EagerSelectorNamespace
from narwhals._pandas_like.expr import PandasLikeExpr

if TYPE_CHECKING:
    from narwhals._pandas_like.dataframe import PandasLikeDataFrame  # noqa: F401
    from narwhals._pandas_like.series import PandasLikeSeries  # noqa: F401


class PandasSelectorNamespace(
    EagerSelectorNamespace["PandasLikeDataFrame", "PandasLikeSeries"]
):
    @property
    def _selector(self) -> type[PandasSelector]:
        return PandasSelector


class PandasSelector(  # type: ignore[misc]
    CompliantSelector["PandasLikeDataFrame", "PandasLikeSeries"], PandasLikeExpr
):
    def _to_expr(self) -> PandasLikeExpr:
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
