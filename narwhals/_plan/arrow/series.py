from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._plan.dummy import DummyCompliantSeries

if TYPE_CHECKING:
    from narwhals._arrow.typing import ChunkedArrayAny  # noqa: F401


class ArrowSeries(DummyCompliantSeries["ChunkedArrayAny"]):
    def to_list(self) -> list[Any]:
        return self.native.to_pylist()
