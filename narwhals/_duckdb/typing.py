from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol

if TYPE_CHECKING:
    from duckdb import Expression

    from narwhals._duckdb.utils import UnorderableWindowInputs
    from narwhals._duckdb.utils import WindowInputs

    class WindowFunction(Protocol):
        def __call__(self, window_inputs: WindowInputs) -> Expression: ...

    class UnorderableWindowFunction(Protocol):
        def __call__(self, window_inputs: UnorderableWindowInputs) -> Expression: ...
