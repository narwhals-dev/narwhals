from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Container
from typing import Iterable
from typing import Literal
from typing import Protocol

from narwhals._compliant.typing import CompliantExprT
from narwhals._compliant.typing import CompliantFrameT
from narwhals._compliant.typing import EagerDataFrameT
from narwhals._compliant.typing import EagerExprT
from narwhals._compliant.typing import EagerSeriesT_co
from narwhals.utils import deprecated

if TYPE_CHECKING:
    from narwhals._compliant.selectors import CompliantSelectorNamespace
    from narwhals.dtypes import DType

__all__ = ["CompliantNamespace", "EagerNamespace"]


class CompliantNamespace(Protocol[CompliantFrameT, CompliantExprT]):
    def col(self, *column_names: str) -> CompliantExprT: ...
    def lit(self, value: Any, dtype: DType | None) -> CompliantExprT: ...
    def exclude(self, excluded_names: Container[str]) -> CompliantExprT: ...
    def nth(self, *column_indices: int) -> CompliantExprT: ...
    def len(self) -> CompliantExprT: ...
    def all(self) -> CompliantExprT: ...
    def all_horizontal(self, *exprs: CompliantExprT) -> CompliantExprT: ...
    def any_horizontal(self, *exprs: CompliantExprT) -> CompliantExprT: ...
    def sum_horizontal(self, *exprs: CompliantExprT) -> CompliantExprT: ...
    def mean_horizontal(self, *exprs: CompliantExprT) -> CompliantExprT: ...
    def min_horizontal(self, *exprs: CompliantExprT) -> CompliantExprT: ...
    def max_horizontal(self, *exprs: CompliantExprT) -> CompliantExprT: ...
    def concat(
        self,
        items: Iterable[CompliantFrameT],
        *,
        how: Literal["horizontal", "vertical", "diagonal"],
    ) -> CompliantFrameT: ...
    def when(self, predicate: CompliantExprT) -> Any: ...
    def concat_str(
        self,
        *exprs: CompliantExprT,
        separator: str,
        ignore_nulls: bool,
    ) -> CompliantExprT: ...
    @property
    def selectors(self) -> CompliantSelectorNamespace[Any, Any]: ...
    @property
    def _expr(self) -> type[CompliantExprT]: ...


class EagerNamespace(
    CompliantNamespace[EagerDataFrameT, EagerExprT],
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
