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

if TYPE_CHECKING:
    from narwhals._compliant.selectors import CompliantSelectorNamespace
    from narwhals.dtypes import DType
    from narwhals.utils import Implementation
    from narwhals.utils import Version

__all__ = ["CompliantNamespace", "EagerNamespace"]


class CompliantNamespace(Protocol[CompliantFrameT, CompliantExprT]):
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version

    def all(self) -> CompliantExprT: ...
    def col(self, *column_names: str) -> CompliantExprT: ...
    def exclude(self, excluded_names: Container[str]) -> CompliantExprT: ...
    def nth(self, *column_indices: int) -> CompliantExprT: ...
    def len(self) -> CompliantExprT: ...
    def lit(self, value: Any, dtype: DType | None) -> CompliantExprT: ...
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
    def _series(self) -> type[EagerSeriesT_co]: ...
