from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from narwhals._plan.compliant.typing import ExprT_co, FrameT_contra

if TYPE_CHECKING:
    from narwhals._plan.expressions import FunctionExpr as FExpr
    from narwhals._plan.expressions.struct import FieldByName


class ExprStructNamespace(Protocol[FrameT_contra, ExprT_co]):
    def field(
        self, node: FExpr[FieldByName], frame: FrameT_contra, name: str
    ) -> ExprT_co: ...
