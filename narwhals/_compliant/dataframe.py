from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterator
from typing import Mapping
from typing import Protocol
from typing import Sequence

from narwhals._compliant.typing import CompliantSeriesT_co

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.dtypes import DType

__all__ = ["CompliantDataFrame", "CompliantLazyFrame"]


class CompliantDataFrame(Protocol[CompliantSeriesT_co]):
    def __narwhals_dataframe__(self) -> Self: ...
    def __narwhals_namespace__(self) -> Any: ...
    def simple_select(
        self, *column_names: str
    ) -> Self: ...  # `select` where all args are column names.
    def aggregate(self, *exprs: Any) -> Self:  # pragma: no cover
        ...  # `select` where all args are aggregations or literals
        # (so, no broadcasting is necessary).

    @property
    def columns(self) -> Sequence[str]: ...
    @property
    def schema(self) -> Mapping[str, DType]: ...
    def get_column(self, name: str) -> CompliantSeriesT_co: ...
    def iter_columns(self) -> Iterator[CompliantSeriesT_co]: ...


class CompliantLazyFrame(Protocol):
    def __narwhals_lazyframe__(self) -> Self: ...
    def __narwhals_namespace__(self) -> Any: ...
    def simple_select(
        self, *column_names: str
    ) -> Self: ...  # `select` where all args are column names.
    def aggregate(self, *exprs: Any) -> Self:  # pragma: no cover
        ...  # `select` where all args are aggregations or literals
        # (so, no broadcasting is necessary).

    @property
    def columns(self) -> Sequence[str]: ...
    @property
    def schema(self) -> Mapping[str, DType]: ...
    def _iter_columns(self) -> Iterator[Any]: ...
