from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol

from narwhals._compliant.typing import CompliantFrameT
from narwhals._compliant.typing import CompliantSeriesOrNativeExprT_co
from narwhals._compliant.typing import EagerDataFrameT
from narwhals._compliant.typing import EagerSeriesT

if TYPE_CHECKING:
    from narwhals._compliant.expr import CompliantExpr
    from narwhals._compliant.expr import EagerExpr
    from narwhals._compliant.selectors import CompliantSelectorNamespace
    from narwhals.dtypes import DType

__all__ = ["CompliantNamespace"]


class CompliantNamespace(Protocol[CompliantFrameT, CompliantSeriesOrNativeExprT_co]):
    def col(
        self, *column_names: str
    ) -> CompliantExpr[CompliantFrameT, CompliantSeriesOrNativeExprT_co]: ...
    def lit(
        self, value: Any, dtype: DType | None
    ) -> CompliantExpr[CompliantFrameT, CompliantSeriesOrNativeExprT_co]: ...
    @property
    def selectors(self) -> CompliantSelectorNamespace[Any, Any]: ...


class EagerNamespace(
    CompliantNamespace[EagerDataFrameT, EagerSeriesT],
    Protocol[EagerDataFrameT, EagerSeriesT],
):
    # NOTE: Supporting moved ops
    # - `self_create_expr_from_callable` -> `self._expr._from_callable`
    # - `self_create_expr_from_series` -> `self._expr._from_series`
    @property
    def _expr(self) -> type[EagerExpr[EagerDataFrameT, EagerSeriesT]]: ...

    # NOTE: Supporting moved ops
    # - `self._create_series_from_scalar` -> `EagerSeries()._from_scalar`
    #   - Was dependent on a `reference_series`, so is now an instance method
    # - `<class>._from_iterable` -> `self._series._from_iterable`
    @property
    def _series(self) -> type[EagerSeriesT]: ...

    def all_horizontal(
        self, *exprs: EagerExpr[EagerDataFrameT, EagerSeriesT]
    ) -> EagerExpr[EagerDataFrameT, EagerSeriesT]: ...
