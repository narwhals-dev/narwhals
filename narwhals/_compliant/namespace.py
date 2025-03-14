from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol

from narwhals._compliant.typing import CompliantFrameT
from narwhals._compliant.typing import CompliantSeriesOrNativeExprT_co
from narwhals._compliant.typing import EagerDataFrameT
from narwhals._compliant.typing import EagerExprT
from narwhals._compliant.typing import EagerSeriesT_co
from narwhals.utils import deprecated

if TYPE_CHECKING:
    from narwhals._compliant.expr import CompliantExpr
    from narwhals._compliant.selectors import CompliantSelectorNamespace
    from narwhals.dtypes import DType

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
    @property
    def _expr(self) -> type[EagerExprT]: ...
    @property
    def _series(self) -> type[EagerSeriesT_co]: ...
    def all_horizontal(self, *exprs: EagerExprT) -> EagerExprT: ...

    @deprecated(
        "Internally used for `numpy.ndarray` -> `CompliantSeries`\n"
        "Also referenced in untyped `nw.dataframe.DataFrame._extract_compliant`\n"
        "See Also:\n"
        "  - https://github.com/narwhals-dev/narwhals/pull/2149#discussion_r1986283345\n"
        "  - https://github.com/narwhals-dev/narwhals/issues/2116\n"
        "  - https://github.com/narwhals-dev/narwhals/pull/2169"
    )
    def _create_compliant_series(self, value: Any) -> EagerSeriesT_co: ...
