from __future__ import annotations

# NOTE: Convenience re-exports
from altair import Axis, Color, Fill, Opacity, SortField, X, Y, binding_range, datasets

# NOTE: APIs that require changes to support narwhals expressions
from narwhals._plan.altair.api.chart import Chart, layer
from narwhals._plan.altair.api.parameter import param, selection_interval, selection_point

__all__ = (
    "Axis",
    "Chart",
    "Color",
    "Fill",
    "Opacity",
    "SortField",
    "X",
    "Y",
    "binding_range",
    "datasets",
    "layer",
    "param",
    "selection_interval",
    "selection_point",
)
