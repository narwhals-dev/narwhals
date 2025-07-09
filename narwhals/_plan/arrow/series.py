from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._arrow.utils import native_to_narwhals_dtype
from narwhals._plan.protocols import DummyCompliantSeries

if TYPE_CHECKING:
    from narwhals._arrow.typing import ChunkedArrayAny  # noqa: F401
    from narwhals.dtypes import DType


class ArrowSeries(DummyCompliantSeries["ChunkedArrayAny"]):
    def to_list(self) -> list[Any]:
        return self.native.to_pylist()

    def __len__(self) -> int:
        return self.native.length()

    @property
    def dtype(self) -> DType:
        return native_to_narwhals_dtype(self.native.type, self._version)
