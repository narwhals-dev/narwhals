"""Adds support for `alt.param(expr: nw.Expr)` and minor typing improvements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import altair as alt
from altair import Undefined
from altair.utils import is_undefined

from narwhals._plan.altair._experimental import parameter as alt_p
from narwhals._plan.altair.expression import parse_into_vega_expr

if TYPE_CHECKING:
    from collections.abc import Sequence

    from altair import theme
    from altair.vegalite.v6.schema._typing import (
        SelectionResolution_T,
        SingleDefUnitChannel_T,
    )

    import narwhals._plan as nw
    from narwhals._plan.altair import typing as alt_t
    from narwhals._plan.altair._experimental import stream
    from narwhals._plan.altair.typing import Optional

__all__ = ["param", "selection_interval", "selection_point"]


def param(
    name: Optional[str] = Undefined,
    *,
    expr: Optional[nw.Expr | alt_t.IntoAltExpr] = Undefined,
    value: Optional[Any] = Undefined,
    bind: Optional[alt.Binding] = Undefined,
    react: Optional[Literal[False]] = Undefined,
) -> alt.Parameter:
    """Create a variable parameter.

    ## See Also
    [`altair.theme.VariableParameterKwds`][]
    """
    if not is_undefined(expr):
        expr = parse_into_vega_expr(expr)

    kwds = alt_p.ensure_param_name(
        _keep_defined(
            name=name,
            expr=expr,
            value=value,
            bind=bind,
            react=react,
            param_type="variable",
        )
    )
    return alt.Parameter(
        name=kwds["name"],
        param_type=kwds.pop("param_type"),
        param=alt.VariableParameter(**kwds),
    )


if TYPE_CHECKING:

    def selection_interval(
        name: str | None = None,
        *,
        value: Optional[alt.api._SelectionIntervalValueMap] = Undefined,
        bind: Optional[alt.Binding] = Undefined,
        empty: Optional[Literal[False]] = Undefined,
        encodings: Optional[Sequence[SingleDefUnitChannel_T]] = Undefined,
        on: Optional[stream.Stream | stream.StreamSelector] = Undefined,
        clear: Optional[
            stream.Stream | stream.StreamSelector | Literal[False]
        ] = Undefined,
        resolve: Optional[SelectionResolution_T] = Undefined,
        mark: Optional[theme.BrushConfigKwds] = Undefined,
        translate: Optional[stream.StreamSelector | Literal[False]] = Undefined,
        zoom: Optional[stream.StreamSelector | Literal[False]] = Undefined,
    ) -> alt.Parameter:
        """Create an interval selection parameter.

        ## See Also
        - [`altair.theme.TopLevelSelectionParameterKwds`][]
        - [`altair.theme.IntervalSelectionConfigKwds`][]
        """
        ...

    def selection_point(
        name: str | None = None,
        *,
        value: Optional[alt.api._SelectionPointValue] = Undefined,
        bind: Optional[alt.Binding] = Undefined,
        empty: Optional[Literal[False]] = Undefined,
        encodings: Optional[Sequence[SingleDefUnitChannel_T]] = Undefined,
        fields: Optional[Sequence[str]] = Undefined,
        on: Optional[stream.Stream | stream.StreamSelector] = Undefined,
        clear: Optional[
            stream.Stream | stream.StreamSelector | Literal[False]
        ] = Undefined,
        resolve: Optional[SelectionResolution_T] = Undefined,
        toggle: Optional[alt_t.VegaExpr | Literal[False]] = Undefined,
        nearest: Optional[Literal[True]] = Undefined,
    ) -> alt.Parameter:
        """Create a point selection parameter.

        ## See Also
        - [`altair.theme.TopLevelSelectionParameterKwds`][]
        - [`altair.theme.PointSelectionConfigKwds`][]
        """
        ...

else:
    selection_interval = alt.selection_interval
    selection_point = alt.selection_point


def _keep_defined(**kwds: Any) -> dict[str, Any]:
    """Too painful to have this working soundly."""
    return {k: v for k, v in kwds.items() if v is not Undefined}
