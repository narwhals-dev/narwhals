from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol

if TYPE_CHECKING:
    from sqlframe.base.column import Column

    from narwhals._spark_like.utils import WindowInputs

    class WindowFunction(Protocol):
        def __call__(self, window_inputs: WindowInputs) -> Column: ...
