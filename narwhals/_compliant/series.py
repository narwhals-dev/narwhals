from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.dtypes import DType

__all__ = ["CompliantSeries"]


class CompliantSeries(Protocol):
    @property
    def dtype(self) -> DType: ...
    @property
    def name(self) -> str: ...
    def __narwhals_series__(self) -> CompliantSeries: ...
    def alias(self, name: str) -> Self: ...
