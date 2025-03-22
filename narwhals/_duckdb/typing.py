from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol
from typing import Sequence

if TYPE_CHECKING:
    import duckdb

    class WindowFunction(Protocol):
        def __call__(
            self,
            _input: duckdb.Expression,
            partition_by: Sequence[str],
            order_by: Sequence[str],
        ) -> duckdb.Expression: ...
