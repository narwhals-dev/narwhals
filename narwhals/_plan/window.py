from __future__ import annotations

from narwhals._plan.common import ExprIR


class Window(ExprIR): ...


class OverWindow(Window): ...


class RollingWindow(Window): ...


class RollingSum(RollingWindow): ...


class RollingMean(RollingWindow): ...


class RollingVar(RollingWindow): ...


class RollingStd(RollingWindow): ...
