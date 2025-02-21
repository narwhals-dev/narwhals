from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING
from typing import Iterator

from narwhals._pandas_like.dataframe import PandasLikeDataFrame
from narwhals._pandas_like.expr import PandasLikeExpr
from narwhals._pandas_like.series import PandasLikeSeries
from narwhals._selectors import CompliantSelector
from narwhals._selectors import CompliantSelectorNamespace

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._pandas_like.dataframe import PandasLikeDataFrame
    from narwhals._pandas_like.series import PandasLikeSeries
    from narwhals._selectors import EvalNames
    from narwhals._selectors import EvalSeries
    from narwhals.utils import _FullContext


class PandasSelectorNamespace(
    CompliantSelectorNamespace["PandasLikeDataFrame", "PandasLikeSeries"]
):
    def _iter_columns(self, df: PandasLikeDataFrame) -> Iterator[PandasLikeSeries]:
        from narwhals._pandas_like.series import PandasLikeSeries

        series = partial(
            PandasLikeSeries,
            implementation=df._implementation,
            backend_version=df._backend_version,
            version=df._version,
        )
        # NOTE: (PERF102) is a false-positive
        # .items() -> (str, pd.Series)
        # .values() -> np.ndarray
        for _col, ser in df._native_frame.items():  # noqa: PERF102
            yield series(ser)

    def _selector(
        self,
        context: _FullContext,
        call: EvalSeries[PandasLikeDataFrame, PandasLikeSeries],
        evaluate_output_names: EvalNames[PandasLikeDataFrame],
        /,
    ) -> CompliantSelector[PandasLikeDataFrame, PandasLikeSeries]:
        return PandasSelector(
            call,
            depth=0,
            function_name="selector",
            evaluate_output_names=evaluate_output_names,
            alias_output_names=None,
            implementation=context._implementation,
            backend_version=context._backend_version,
            version=context._version,
        )

    def __init__(self: Self, context: _FullContext, /) -> None:
        self._implementation = context._implementation
        self._backend_version = context._backend_version
        self._version = context._version


class PandasSelector(  # type: ignore[misc]
    CompliantSelector["PandasLikeDataFrame", "PandasLikeSeries"], PandasLikeExpr
):
    @property
    def selectors(self) -> PandasSelectorNamespace:
        return PandasSelectorNamespace(self)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PandasSelector(depth={self._depth}, function_name={self._function_name}, "
        )

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
