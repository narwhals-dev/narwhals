from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    import ibis.expr.types as ir

    from narwhals._ibis.utils import WindowInputs

    ExprT = TypeVar("ExprT", bound=ir.Value)

    class WindowFunction(Protocol):
        def __call__(self, window_inputs: WindowInputs) -> ir.Value: ...
