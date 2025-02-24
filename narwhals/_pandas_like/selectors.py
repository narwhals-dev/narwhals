from __future__ import annotations

import sys
from functools import partial
from typing import TYPE_CHECKING
from typing import Any
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
        for _col, ser in df._native_frame.items():  # noqa: PERF102
            yield series(ser)

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
            implementation=self._implementation,  # AttributeError: 'PandasSelector' object has no attribute '_implementation'
            backend_version=self._backend_version,
            version=self._version,
        )

    def __init__(self: Self, context: _FullContext, /) -> None:
        self._implementation = context._implementation
        self._backend_version = context._backend_version
        self._version = context._version


# BUG: `3.8` Protocol?
# https://github.com/narwhals-dev/narwhals/pull/2064#discussion_r1965980715
class PandasSelector(  # type: ignore[misc]
    CompliantSelector["PandasLikeDataFrame", "PandasLikeSeries"], PandasLikeExpr
):
    if sys.version_info < (3, 9):

        def __init__(self, *args: Any, **kwds: Any) -> None:
            super(PandasLikeExpr).__init__(*args, **kwds)

    def _to_expr(self: Self) -> PandasLikeExpr:
        return PandasLikeExpr(
            self._call,  # AttributeError: 'PandasSelector' object has no attribute '_call'
            depth=self._depth,
            function_name=self._function_name,
            evaluate_output_names=self._evaluate_output_names,  # AttributeError: 'PandasSelector' object has no attribute '_evaluate_output_names'
            alias_output_names=self._alias_output_names,
            implementation=self._implementation,  # AttributeError: 'PandasSelector' object has no attribute '_implementation'
            backend_version=self._backend_version,
            version=self._version,
        )
