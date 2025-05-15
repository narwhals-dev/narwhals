"""TODO: Attributes."""

from __future__ import annotations

from narwhals._plan.common import ExprIR


class Window(ExprIR):
    """Renamed from `WindowType`.

    https://github.com/pola-rs/polars/blob/112cab39380d8bdb82c6b76b31aca9b58c98fd93/crates/polars-plan/src/dsl/options/mod.rs#L139
    """


class OverWindow(Window): ...


class RollingWindow(Window): ...


class RollingSum(RollingWindow): ...


class RollingMean(RollingWindow): ...


class RollingVar(RollingWindow): ...


class RollingStd(RollingWindow): ...
