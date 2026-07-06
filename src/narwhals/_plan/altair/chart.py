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
from narwhals._plan.altair import encode
from narwhals._plan.altair.aggregate import aggregate_transform, window_transform
from narwhals._plan.altair.calculate import calculate_transform

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from _typeshed import Incomplete
    from altair.typing import ChartType as AltChart, Optional
    from altair.vegalite.v6.schema._config import ThemeConfig as _ChartKwds
    from altair.vegalite.v6.schema._typing import StackOffset_T
    from altair.vegalite.v6.schema.mixins import _MarkDef
    from typing_extensions import Self, Unpack

    from narwhals._plan.altair.typing import EncodeKwds, FieldName, IntoAltExpr

_EMPTY_SCHEMA: Final = stable_v1.Schema()


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
        """Add named window aggregations to the chart.

        ## Chart ideas
        - [first,last](https://github.com/vega/altair/blob/48b388f140c79d29056d6ea56e519b27e2ed8838/tests/examples_methods_syntax/co2_concentration.py#L23-L29)
        """
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
        """Add named aggregations to the chart.

        ## Chart ideas
        - [`argmax` -> use `struct.field` in encode?](https://github.com/vega/altair/blob/48b388f140c79d29056d6ea56e519b27e2ed8838/tests/examples_methods_syntax/line_chart_with_custom_legend.py#L25-L31)
        """
        return self._from_altair(
            self._chart._add_transform(
                *aggregate_transform(*exprs, group_by=groupby, **named_exprs)
            )
        )

    def transform_filter(
        self,
        predicate: Optional[nw.Expr | Incomplete] = alt.Undefined,
        *more_predicates: nw.Expr | Incomplete,
        empty: Optional[bool] = alt.Undefined,
        **constraints: nw.Expr | Incomplete,
    ) -> Self:
        """Add a filter to the chart.

        ## Chart ideas
        - [2x `FieldOneOfPredicate` 1](https://github.com/vega/altair/blob/48b388f140c79d29056d6ea56e519b27e2ed8838/tests/examples_methods_syntax/ranged_dot_plot.py#L15-L21)
        - [2x `FieldOneOfPredicate` 2](https://github.com/vega/altair/blob/48b388f140c79d29056d6ea56e519b27e2ed8838/tests/examples_methods_syntax/line_chart_with_cumsum_faceted.py#L14-L19)
        - [calculate_residuals](https://github.com/vega/altair/blob/48b388f140c79d29056d6ea56e519b27e2ed8838/tests/examples_methods_syntax/calculate_residuals.py#L14-L32)
            - 2x filter: is_not_null + is_between/just use less than?
            - Also has some weird renaming on calculate that could be cleaner with positional + alias
        - [interactive_column_selection](https://github.com/vega/altair/blob/48b388f140c79d29056d6ea56e519b27e2ed8838/tests/examples_methods_syntax/interactive_column_selection.py#L56-L59)
            - good candidate for complexity
            - but heavy parameter usage and pandas indexing
        - [all the features](https://github.com/vega/altair/blob/48b388f140c79d29056d6ea56e519b27e2ed8838/tests/examples_methods_syntax/multiple_interactions.py#L20-L33)
        """
        msg = "TODO @dangotbanned: `transform_filter`"
        raise NotImplementedError(msg)

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

    def encode(self, *args: nw.Expr | Any, **kwds: Unpack[EncodeKwds]) -> Self:
        """Map properties of the data to visual properties of the chart.

        ## Chart ideas:
        - [argmax rewrite](https://github.com/vega/altair/blob/48b388f140c79d29056d6ea56e519b27e2ed8838/tests/examples_methods_syntax/line_with_last_value_labeled.py#L26-L31)
        - [conditional within a property setter](https://github.com/vega/altair/blob/48b388f140c79d29056d6ea56e519b27e2ed8838/tests/examples_methods_syntax/lasagna_plot.py#L11-L32)
        """
        from_expr = encode.from_expr
        args_ = (from_expr(e._ir, self) if isinstance(e, nw.Expr) else e for e in args)
        kwds_ = {
            # TODO @dangotbanned: review if `mypy` understands this after bump for PEP 728
            channel: (from_expr(e._ir, self, channel) if isinstance(e, nw.Expr) else e)  # type: ignore[call-overload]
            for channel, e in kwds.items()
        }
        return self._from_altair(self._chart.encode(*args_, **kwds_))  # type: ignore[arg-type]

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
        def mark_point(self, **kwds: Any) -> Self: ...

        mark_arc = mark_area = mark_bar = mark_circle = mark_geoshape = mark_image = (
            mark_line
        ) = mark_rect = mark_rule = mark_square = mark_text = mark_tick = mark_trail = (
            mark_point
        )

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
        return Chart._from_altair(alt.LayerChart(layer=(self._chart, other._chart)))

    def __and__(self, other: Chart) -> Chart:
        return Chart._from_altair(alt.VConcatChart(vconcat=(self._chart, other._chart)))

    def __or__(self, other: Chart) -> Chart:
        # `TopLevelMixin.__or__` has some logic for `Concat` vs `HConcat`
        return Chart._from_altair(self._chart.__or__(other._chart))


def layer(*charts: Chart, **kwds: Unpack[_ChartKwds]) -> Chart:
    """Layer multiple charts."""
    return Chart._from_altair(
        alt.LayerChart(layer=tuple(c._chart for c in charts), **kwds)
    )


def _wrapper(chart: Chart, method_name: str) -> Callable[..., Chart]:
    def _(*args: Any, **kwds: Any) -> Chart:
        native = getattr(chart._chart, method_name)(*args, **kwds)
        return chart._from_altair(native)

    return _
