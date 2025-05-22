from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sqlframe.base.column import Column

    from narwhals._spark_like.utils import UnorderableWindowInputs, WindowInputs

    class WindowFunction(Protocol):
        def __call__(self, window_inputs: WindowInputs, /) -> Column: ...

    class UnorderableWindowFunction(Protocol):
        def __call__(self, window_inputs: UnorderableWindowInputs, /) -> Column: ...
