"""A thin wrapper around `altair.Chart` to demo expression support.

A real integration would update these APIs to expect `narwhals._plan.Expr` objects, but this'll do for now.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import altair as alt
import altair.utils

import narwhals._plan as nw
from narwhals._plan import expressions as ir
from narwhals._plan.altair.aggregate import window_transform
from narwhals._plan.altair.calculate import calculate_transform
from narwhals._plan.altair.conditional import (
    ConditionalField,
    ConditionalValue,
    encode_ternary_expr,
)
from narwhals._plan.altair.exceptions import unsupported_error

if TYPE_CHECKING:
    from collections.abc import Callable

    from altair.typing import ChartType as AltChart
    from altair.vegalite.v6.schema._config import ThemeConfig as _ChartKwds
    from altair.vegalite.v6.schema.mixins import _MarkDef
    from typing_extensions import Self, Unpack

    from narwhals._plan.altair.typing import EncodeKwds, IntoAltExpr


class Chart:
    def __init__(
        self, data: alt.ChartDataType = alt.Undefined, /, **kwds: Unpack[_ChartKwds]
    ) -> None:
        # TODO @dangotbanned: Widen `height`, `width` to use `Map` (not `dict[str, Any]`)
        self._chart: AltChart = alt.Chart(data, **kwds)  # type: ignore[arg-type]

    @classmethod
    def _from_altair(cls, chart: AltChart, /) -> Self:
        self = cls.__new__(cls)
        self._chart = chart
        return self

    def transform_calculate(
        self, *exprs: nw.Expr, **named_exprs: nw.Expr | IntoAltExpr
    ) -> Self:
        """Add named expressions to the chart.

        Tip:
            This guy is similar to `with_columns`.
        """
        return self._from_altair(
            self._chart._add_transform(*calculate_transform(*exprs, **named_exprs))
        )

    # TODO @dangotbanned: Need to accept and pass-through non-Expr inputs
    def transform_window(self, **named_exprs: nw.Expr) -> Self:
        return self._from_altair(
            self._chart._add_transform(window_transform(**named_exprs))
        )

    def encode(self, *args: nw.Expr | Any, **kwds: Unpack[EncodeKwds]) -> Self:
        args_ = (_encode_expr(e) if isinstance(e, nw.Expr) else e for e in args)
        kwds_ = {
            channel: (_encode_expr(e) if isinstance(e, nw.Expr) else e)
            for channel, e in kwds.items()
        }
        return self._from_altair(self._chart.encode(*args_, **kwds_))  # type: ignore[arg-type]

    def to_altair(self) -> AltChart:
        return self._chart

    def _repr_mimebundle_(self, *args: Any, **kwds: Any) -> Any:
        return self._chart._repr_mimebundle_(*args, **kwds)

    def __repr__(self) -> str:
        return self._chart.__repr__()

    if TYPE_CHECKING:

        @altair.utils.use_signature(_MarkDef)
        def mark_bar(self, **kwds: Any) -> Self: ...
        @altair.utils.use_signature(_MarkDef)
        def mark_rule(self, **kwds: Any) -> Self: ...
        @altair.utils.use_signature(_MarkDef)
        def mark_text(self, **kwds: Any) -> Self: ...
        @altair.utils.use_signature(_MarkDef)
        def mark_line(self, **kwds: Any) -> Self: ...
        @altair.utils.use_signature(_MarkDef)
        def mark_point(self, **kwds: Any) -> Self: ...
        @altair.utils.use_signature(_MarkDef)
        def mark_area(self, **kwds: Any) -> Self: ...
        @altair.utils.use_signature(_MarkDef)
        def mark_rect(self, **kwds: Any) -> Self: ...
        @altair.utils.use_signature(_MarkDef)
        def mark_tick(self, **kwds: Any) -> Self: ...
        @altair.utils.use_signature(_MarkDef)
        def mark_trail(self, **kwds: Any) -> Self: ...
        @altair.utils.use_signature(_MarkDef)
        def mark_arc(self, **kwds: Any) -> Self: ...

    else:

        def __getattr__(self, name: str) -> Callable[..., Chart]:
            if not name.startswith("mark_"):
                msg = f"{type(self).__name__!r} object has no attribute {name!r}"
                raise AttributeError(msg)
            return _wrapper(self, name)


def layer(*charts: Chart, **kwds: Unpack[_ChartKwds]) -> Chart:
    """Layer multiple charts."""
    return Chart._from_altair(
        alt.LayerChart(layer=tuple(c._chart for c in charts), **kwds)
    )


# TODO @dangotbanned: Add more cases as they show up in other examples
# - lit -> value
# - aggregate -> aggregate_field_def
def _encode_expr(expr: nw.Expr) -> ConditionalValue | ConditionalField:
    e = expr._ir
    if isinstance(e, ir.TernaryExpr):
        return encode_ternary_expr(e)
    raise unsupported_error(e, "encoding")


def _wrapper(chart: Chart, method_name: str) -> Callable[..., Chart]:
    def _(*args: Any, **kwds: Any) -> Chart:
        native = getattr(chart._chart, method_name)(*args, **kwds)
        return chart._from_altair(native)

    return _
