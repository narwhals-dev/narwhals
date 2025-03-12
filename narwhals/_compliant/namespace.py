from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol

from narwhals._compliant.typing import CompliantFrameT
from narwhals._compliant.typing import CompliantSeriesOrNativeExprT_co
from narwhals._compliant.typing import EagerDataFrameT
from narwhals._compliant.typing import EagerExprT
from narwhals._compliant.typing import EagerSeriesT_co

if TYPE_CHECKING:
    from narwhals._compliant.expr import CompliantExpr
    from narwhals._compliant.selectors import CompliantSelectorNamespace
    from narwhals.dtypes import DType
    from narwhals.utils import Implementation
    from narwhals.utils import Version

__all__ = ["CompliantNamespace", "EagerNamespace"]


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
    CompliantNamespace[EagerDataFrameT, EagerSeriesT_co],
    Protocol[EagerDataFrameT, EagerSeriesT_co, EagerExprT],
):
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version

    @property
    def _expr(self) -> type[EagerExprT]: ...
    @property
    def _series(self) -> type[EagerSeriesT_co]: ...
    def all_horizontal(self, *exprs: EagerExprT) -> EagerExprT: ...
