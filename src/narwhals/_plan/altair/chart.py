"""A thin wrapper around `altair.Chart` to demo expression support.

A real integration would update these APIs to expect `narwhals._plan.Expr` objects, but this'll do for now.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Final

import altair as alt
import altair.utils

import narwhals._plan as nw
import narwhals.stable.v1 as stable_v1
from narwhals._plan import expressions as ir
from narwhals._plan.altair.aggregate import aggregate_transform, window_transform
from narwhals._plan.altair.calculate import calculate_transform
from narwhals._plan.altair.conditional import (
    ConditionalField,
    ConditionalValue,
    _value,
    encode_ternary_expr,
)
from narwhals._plan.altair.exceptions import unsupported_error

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from altair.typing import ChartType as AltChart, Optional
    from altair.vegalite.v6.schema._config import ThemeConfig as _ChartKwds
    from altair.vegalite.v6.schema._typing import StackOffset_T
    from altair.vegalite.v6.schema.mixins import _MarkDef
    from typing_extensions import Self, Unpack

    from narwhals._plan.altair.typing import (
        EncodeKwds,
        Field,
        FieldName,
        IntoAltExpr,
        Value,
        VegaType,
    )
    from narwhals.dtypes import DType

_EMPTY_SCHEMA: Final = stable_v1.Schema()

_SECONDARY_FIELD: Final = frozenset(
    (
        "latitude2",
        "longitude2",
        "radius2",
        "theta2",
        "x2",
        "xError",
        "xError2",
        "y2",
        "yError",
        "yError2",
    )
)
"""Channels that don't accept a `type`."""


class Chart:
    def __init__(
        self, data: alt.ChartDataType = alt.Undefined, /, **kwds: Unpack[_ChartKwds]
    ) -> None:
        # TODO @dangotbanned: Widen `height`, `width` to use `Map` (not `dict[str, Any]`)
        maybe_frame = stable_v1.from_native(data, pass_through=True)
        self._chart: AltChart = alt.Chart(maybe_frame, **kwds)  # type: ignore[arg-type]

    @classmethod
    def _from_altair(cls, chart: AltChart, /) -> Self:
        self = cls.__new__(cls)
        self._chart = chart
        return self

    # TODO @dangotbanned: Support non-string literals in `**named_exprs`
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
    def transform_window(
        self,
        frame: Optional[Sequence[float | None]] = alt.Undefined,
        groupby: Optional[Sequence[FieldName]] = alt.Undefined,
        sort: Optional[Sequence[alt.SortField | dict[str, str]]] = alt.Undefined,
        **named_exprs: nw.Expr,
    ) -> Self:
        return self._from_altair(
            self._chart._add_transform(
                window_transform(frame=frame, group_by=groupby, sort=sort, **named_exprs)
            )
        )

    def transform_aggregate(
        self,
        *exprs: nw.Expr,
        groupby: Optional[Sequence[FieldName]] = alt.Undefined,
        **named_exprs: nw.Expr,
    ) -> Self:
        return self._from_altair(
            self._chart._add_transform(
                *aggregate_transform(*exprs, group_by=groupby, **named_exprs)
            )
        )

    # TODO @dangotbanned: Is there a reasonable parallel to lean on here?
    def transform_stack(
        self,
        as_: FieldName | Sequence[FieldName],
        stack: FieldName,
        groupby: Sequence[FieldName] = (),
        offset: Optional[StackOffset_T] = alt.Undefined,
        sort: Optional[Sequence[alt.SortField]] = alt.Undefined,
    ) -> Self:
        """https://vega.github.io/vega/docs/transforms/stack/.

        Not sure if this corresponds to any relational operators.
        """
        return self._from_altair(
            self._chart._add_transform(
                alt.StackTransform(
                    stack=stack, groupby=groupby, offset=offset, sort=sort, **{"as": as_}
                )
            )
        )

    @functools.cached_property
    def _try_collect_schema(self) -> stable_v1.Schema:
        """Collect and cache the schema of a narwhals dataframe.

        ## Notes
        - Persisted for a single `encode` context
        - No-op if the current chart isn't wrapping a dataframe
        - Deferred until an encoding channel requires a type
        - native -> narwhals is cached within narwhals
        - narwhals -> vega is cached here
        - Bypasses `utils.parse_shorthand`, which does a lot more work than this
        """
        if isinstance(self._chart.data, stable_v1.DataFrame):
            # TODO @dangotbanned: Fix this upstream, `stable.v*` needs to override `schema`, `collect_schema`
            return self._chart.data.collect_schema()  # type: ignore[return-value]
        return _EMPTY_SCHEMA

    # TODO @dangotbanned: Add more cases as they show up in other examples
    # - aggregate -> aggregate_field_def
    def encode(self, *args: nw.Expr | Any, **kwds: Unpack[EncodeKwds]) -> Self:
        """Map properties of the data to visual properties of the chart."""
        args_ = (self._encode_expr(e) if isinstance(e, nw.Expr) else e for e in args)
        kwds_ = {
            channel: (self._encode_expr(e, channel) if isinstance(e, nw.Expr) else e)
            for channel, e in kwds.items()
        }
        return self._from_altair(self._chart.encode(*args_, **kwds_))  # type: ignore[arg-type]

    def _encode_expr(
        self, expr: nw.Expr, channel: str | None = None
    ) -> ConditionalValue | ConditionalField | Field | Value:
        e = expr._ir
        if isinstance(e, ir.TernaryExpr):
            return encode_ternary_expr(e)

        if isinstance(e, ir.Column):
            field = e.name
            if channel and channel in _SECONDARY_FIELD:
                return {"field": field, "aggregate": alt.Undefined}
            if dtype := self._try_collect_schema.get(field):
                vtype = _vegalite_type(dtype)
            else:
                vtype = alt.Undefined
            return {"field": field, "type": vtype, "aggregate": alt.Undefined}

        # TODO @dangotbanned: Still unsure when datum/value should be preferred
        if isinstance(e, ir.Lit):
            return {"value": _value(e)}

        raise unsupported_error(e, "encoding")

    def properties(self, **kwds: Unpack[_ChartKwds]) -> Self:
        """Set top-level properties of the chart."""
        return self._from_altair(self._chart.properties(**kwds))

    def to_altair(self) -> AltChart:
        return self._chart

    def _repr_mimebundle_(self, *args: Any, **kwds: Any) -> Any:
        return self._chart._repr_mimebundle_(*args, **kwds)

    def __repr__(self) -> str:
        return self._chart.__repr__()

    if TYPE_CHECKING:

        @altair.utils.use_signature(_MarkDef)
        def mark_bar(self, **kwds: Any) -> Self: ...
        # TODO @dangotbanned: Try writing the rest as aliases?
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

        # NOTE: May be able to write most of the others as aliases
        @altair.utils.use_signature(alt.AxisConfig)
        def configure_axis(self, **kwds: Any) -> Self: ...
        @altair.utils.use_signature(alt.ViewConfig)
        def configure_view(self, **kwds: Any) -> Self: ...
        @altair.utils.use_signature(alt.CompositionConfig)
        def configure_concat(self, **kwds: Any) -> Self: ...

        @altair.utils.use_signature(alt.AxisResolveMap)
        def resolve_axis(self, **kwds: Any) -> Self: ...
        @altair.utils.use_signature(alt.LegendResolveMap)
        def resolve_legend(self, **kwds: Any) -> Self: ...
        @altair.utils.use_signature(alt.ScaleResolveMap)
        def resolve_scale(self, **kwds: Any) -> Self: ...

    else:

        def __getattr__(self, name: str) -> Callable[..., Chart]:
            if not name.startswith(("mark_", "configure", "resolve_")):
                msg = f"{type(self).__name__!r} object has no attribute {name!r}"
                raise AttributeError(msg)
            return _wrapper(self, name)

    def __add__(self, other: Chart) -> Chart:
        return Chart._from_altair(alt.LayerChart((self._chart, other._chart)))

    def __and__(self, other: Chart) -> Chart:
        return Chart._from_altair(alt.VConcatChart((self._chart, other._chart)))

    def __or__(self, other: Chart) -> Chart:
        # `TopLevelMixin.__or__` has some logic for `Concat` vs `HConcat`
        return Chart._from_altair(self._chart.__or__(other._chart))


def layer(*charts: Chart, **kwds: Unpack[_ChartKwds]) -> Chart:
    """Layer multiple charts."""
    return Chart._from_altair(
        alt.LayerChart(layer=tuple(c._chart for c in charts), **kwds)
    )


@functools.lru_cache(16)
def _vegalite_type(dtype: DType, /) -> Optional[VegaType]:
    if dtype.is_numeric():
        return "quantitative"
    if isinstance(dtype, (stable_v1.String, stable_v1.Categorical, stable_v1.Boolean)):
        return "nominal"
    if isinstance(dtype, (stable_v1.Datetime, stable_v1.Date)):
        return "temporal"
    return alt.Undefined


def _wrapper(chart: Chart, method_name: str) -> Callable[..., Chart]:
    def _(*args: Any, **kwds: Any) -> Chart:
        native = getattr(chart._chart, method_name)(*args, **kwds)
        return chart._from_altair(native)

    return _
