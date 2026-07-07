"""Adds support for `alt.param(expr: nw.Expr)` and wrapping altair paramters in `nw.Expr`."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import altair as alt
from altair import Undefined
from altair.utils import is_undefined

from narwhals._plan.altair._experimental.serde import serialize
from narwhals._plan.altair.api import _parameter_ir
from narwhals._plan.altair.api.expression import parse_into_vega_expr

if TYPE_CHECKING:
    from collections.abc import Sequence

    from _typeshed import SupportsItemAccess
    from altair import theme
    from altair.vegalite.v6.schema._typing import (
        SelectionResolution_T,
        SingleDefUnitChannel_T,
    )

    from narwhals._plan.altair._experimental import stream
    from narwhals._plan.altair.api import typing as alt_t
    from narwhals._plan.altair.api.typing import Optional
    from narwhals._plan.expr import Expr as NwExpr

    _KwdsT = TypeVar("_KwdsT", bound=SupportsItemAccess[Any, Any])

__all__ = ["param", "selection_interval", "selection_point"]


def param(
    name: Optional[str] = Undefined,
    *,
    expr: Optional[NwExpr | alt_t.IntoAltExpr] = Undefined,
    value: Optional[Any] = Undefined,
    bind: Optional[alt.Binding] = Undefined,
    react: Optional[Literal[False]] = Undefined,
) -> NwExpr:
    """Create a variable parameter.

    ## See Also
    [`altair.theme.VariableParameterKwds`][]
    """
    if not is_undefined(expr):
        expr = parse_into_vega_expr(expr)

    kwds = ensure_param_name(
        _keep_defined(
            name=name,
            expr=expr,
            value=value,
            bind=bind,
            react=react,
            param_type="variable",
        )
    )
    return _parameter_ir.from_altair(
        alt.Parameter(
            name=kwds["name"],
            param_type=kwds.pop("param_type"),
            param=alt.VariableParameter(**kwds),
        )
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
    ) -> NwExpr:
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
    ) -> NwExpr:
        """Create a point selection parameter.

        ## See Also
        - [`altair.theme.TopLevelSelectionParameterKwds`][]
        - [`altair.theme.PointSelectionConfigKwds`][]
        """
        ...

else:

    def selection_interval(*args: Any, **kwds: Any) -> NwExpr:
        return _parameter_ir.from_altair(alt.selection_interval(*args, **kwds))

    def selection_point(*args: Any, **kwds: Any) -> NwExpr:
        return _parameter_ir.from_altair(alt.selection_point(*args, **kwds))


def _keep_defined(**kwds: Any) -> dict[str, Any]:
    """Too painful to have this working soundly."""
    return {k: v for k, v in kwds.items() if v is not Undefined}


def ensure_param_name(kwds: _KwdsT) -> _KwdsT:
    """Generate a parameter name if we haven't got one yet."""
    if "name" not in kwds:
        encoded = serialize(kwds, deterministic=True, default=str)
        # NOTE: https://github.com/vega/altair/pull/3291#issuecomment-1866999185
        # - 256 vs 224 -> 64 vs 56 characters (only need 16)
        # - not used for security
        name = f"param_{hashlib.sha224(encoded, usedforsecurity=False).hexdigest()[:16]}"
        kwds["name"] = name
    return kwds
