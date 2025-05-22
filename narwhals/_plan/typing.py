from __future__ import annotations

import typing as t

from narwhals._typing_compat import TypeVar

if t.TYPE_CHECKING:
    from narwhals._plan import operators as ops
    from narwhals._plan.common import ExprIR, Function
    from narwhals._plan.functions import RollingWindow

__all__ = ["FunctionT", "LeftT", "OperatorT", "RightT", "RollingT"]


FunctionT = TypeVar("FunctionT", bound="Function")
RollingT = TypeVar("RollingT", bound="RollingWindow")
LeftT = TypeVar("LeftT", bound="ExprIR", default="ExprIR")
OperatorT = TypeVar("OperatorT", bound="ops.Operator", default="ops.Operator")
RightT = TypeVar("RightT", bound="ExprIR", default="ExprIR")
