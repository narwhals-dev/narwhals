from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol

if TYPE_CHECKING:
    import duckdb

    from narwhals._duckdb.utils import WindowInputs

    class WindowFunction(Protocol):
        def __call__(self, window_inputs: WindowInputs) -> duckdb.Expression: ...
