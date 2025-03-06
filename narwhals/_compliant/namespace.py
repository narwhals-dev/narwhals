from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol

from narwhals._compliant.typing import CompliantFrameT
from narwhals._compliant.typing import CompliantSeriesOrNativeExprT_co

if TYPE_CHECKING:
    from narwhals._compliant.expr import CompliantExpr
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
