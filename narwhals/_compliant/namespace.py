from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING
from typing import Any
from typing import Container
from typing import Iterable
from typing import Literal
from typing import Protocol

from narwhals._compliant.typing import CompliantExprT
from narwhals._compliant.typing import CompliantFrameT
from narwhals._compliant.typing import DepthTrackingExprT
from narwhals._compliant.typing import EagerDataFrameT
from narwhals._compliant.typing import EagerExprT
from narwhals._compliant.typing import EagerSeriesT
from narwhals.utils import exclude_column_names
from narwhals.utils import get_column_names
from narwhals.utils import passthrough_column_names

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals._compliant.selectors import CompliantSelectorNamespace
    from narwhals._compliant.when_then import CompliantWhen
    from narwhals._compliant.when_then import EagerWhen
    from narwhals.dtypes import DType
    from narwhals.utils import Implementation
    from narwhals.utils import Version

    Incomplete: TypeAlias = Any

__all__ = ["CompliantNamespace", "EagerNamespace"]


class CompliantNamespace(Protocol[CompliantFrameT, CompliantExprT]):
    _implementation: Implementation
    _backend_version: tuple[int, ...]
    _version: Version

    def all(self) -> CompliantExprT:
        return self._expr.from_column_names(get_column_names, context=self)

    def col(self, *column_names: str) -> CompliantExprT:
        return self._expr.from_column_names(
            passthrough_column_names(column_names), context=self
        )

    def exclude(self, excluded_names: Container[str]) -> CompliantExprT:
        return self._expr.from_column_names(
            partial(exclude_column_names, names=excluded_names), context=self
        )

    def nth(self, *column_indices: int) -> CompliantExprT:
        return self._expr.from_column_indices(*column_indices, context=self)

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
    def when(
        self, predicate: CompliantExprT
    ) -> CompliantWhen[CompliantFrameT, Incomplete, CompliantExprT]: ...
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


class DepthTrackingNamespace(
    CompliantNamespace[CompliantFrameT, DepthTrackingExprT],
    Protocol[CompliantFrameT, DepthTrackingExprT],
):
    def all(self) -> DepthTrackingExprT:
        return self._expr.from_column_names(
            get_column_names, function_name="all", context=self
        )

    def col(self, *column_names: str) -> DepthTrackingExprT:
        return self._expr.from_column_names(
            passthrough_column_names(column_names), function_name="col", context=self
        )

    def exclude(self, excluded_names: Container[str]) -> DepthTrackingExprT:
        return self._expr.from_column_names(
            partial(exclude_column_names, names=excluded_names),
            function_name="exclude",
            context=self,
        )


class EagerNamespace(
    DepthTrackingNamespace[EagerDataFrameT, EagerExprT],
    Protocol[EagerDataFrameT, EagerSeriesT, EagerExprT],
):
    @property
    def _series(self) -> type[EagerSeriesT]: ...
    def when(
        self, predicate: EagerExprT
    ) -> EagerWhen[EagerDataFrameT, EagerSeriesT, EagerExprT, Incomplete]: ...
