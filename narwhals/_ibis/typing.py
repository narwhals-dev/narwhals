from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol

if TYPE_CHECKING:
    import ibis.expr.types as ir

    from narwhals._ibis.utils import WindowInputs

    class WindowFunction(Protocol):
        def __call__(self, window_inputs: WindowInputs) -> ir.Expr: ...
