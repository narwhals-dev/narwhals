"""Trying to make sense of `bind*`, `param`, `selection_*` apis.

Interactions are used in many of the complex examples, so I want to show those too!

## Notes
- `bind*`
    - Returns an object only accepted in `*Parameter.bind`
    - Always need to call two functions to use it
- `param`, `selection_*`
    - Have a large amount of indirection and deprecation code
    - `v6` introduced (https://github.com/vega/altair/pull/3851)
- `selection_*`
    - Defined by `select`
        - but the `select` argument is entirely hidden
    - These APIs accept `expr` and then silently drop it

- `param` (variable)
    - Defined by `expr`, but only used in once in user guide
"""

from __future__ import annotations

# mypy: disable-error-code="typeddict-item"
# > Non-required keys (...) not explicitly found in any ** item"
# TODO @dangotbanned: Review after PEP 728 support
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Final,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeVar,
)

import altair as alt
from altair import Undefined, theme
from altair.theme import VariableParameterKwds as VariableParamKwds
from altair.utils import is_undefined

from narwhals._plan.altair.api.expression import parse_into_vega_expr

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import typing_extensions as te
    from altair.vegalite.v6.schema._typing import (
        SelectionResolution_T,
        SingleDefUnitChannel_T,
    )
    from typing_extensions import Required, Self

    import narwhals._plan as nw
    from narwhals._plan.altair._experimental import stream
    from narwhals._plan.altair.api import typing as alt_t
    from narwhals._plan.altair.api.typing import Optional

    class _CommonBind(te.TypedDict, total=False, closed=True):  # type: ignore[call-arg]
        debounce: float
        element: str
        name: str

    class _CommonParam(TypedDict, total=False, closed=True):  # type: ignore[call-arg]
        """Initial state for a param that could be either variable or selection."""

        name: str
        """If not provided, must be computed as the final build step."""
        bind: Binding  # selection is wider, but legend/scales are one-way (also never seen stream stuff used)
        value: Any


class _CommonParamOpen(TypedDict, total=False):
    """Initial state for a param that could be either variable or selection."""

    name: str
    bind: Binding
    value: Any


_TD_co = TypeVar("_TD_co", bound=_CommonParamOpen, covariant=True)

T = TypeVar("T")

Cloned: TypeAlias = Annotated[T, Literal["cloned"]]
"""Marker to indicate that `T` *should be* a deep-copy.

Important:
    The strategy is, if you see `Cloned` then pass in a copy of yourself.
"""


Binding: TypeAlias = (
    theme.BindCheckboxKwds
    | theme.BindDirectKwds
    | theme.BindInputKwds
    | theme.BindRadioSelectKwds
    | theme.BindRangeKwds
)
"""Anything accepted by `{param,selection_*}(bind=...)`.

Note:
    A TypedDict, sum-type equivalent of `altair.Binding`.
"""


class _SelectionConfig(TypedDict, total=False):
    clear: stream.Stream | stream.StreamSelector | Literal[False]
    encodings: Sequence[SingleDefUnitChannel_T]
    on: stream.Stream | stream.StreamSelector
    resolve: SelectionResolution_T


class IntervalConfigKwds(_SelectionConfig, TypedDict, total=False):
    type: Required[Literal["interval"]]
    mark: theme.BrushConfigKwds
    translate: stream.StreamSelector | Literal[False]
    zoom: stream.StreamSelector | Literal[False]


class PointConfigKwds(_SelectionConfig, TypedDict, total=False):
    type: Required[Literal["point"]]
    fields: Sequence[str]
    nearest: Literal[True]
    toggle: alt_t.VegaExpr | Literal[False]


class _SelectionKwds(_CommonParamOpen, TypedDict, total=False):
    select: Required[_SelectionConfig]
    views: Sequence[str]


class _IntervalKwds(_SelectionKwds, TypedDict, total=False):
    select: Required[IntervalConfigKwds]  # type: ignore[misc]
    value: alt.api._SelectionIntervalValueMap  # type: ignore[misc]


class _PointKwds(_SelectionKwds, TypedDict, total=False):
    select: Required[PointConfigKwds]  # type: ignore[misc]
    value: alt.api._SelectionPointValue  # type: ignore[misc]


_SelectionKwdsT = TypeVar(  # noqa: PLC0105
    "_SelectionKwdsT", bound=_IntervalKwds | _PointKwds, covariant=True
)


_ElementType: TypeAlias = Literal[
    "button",
    "color",
    "date",
    "datetime-local",
    "month",
    "number",
    "search",
    "submit",
    "text",
    "time",
    "week",
]
"""A [HTML form input type] that does not have a dedicated `Bind*` wrapper.

`"search"` is demonstrated in [here](https://altair-viz.github.io/gallery/scatter_point_paths_hover.html)

[HTML form input type]: https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input
"""


class _WithBinding(Protocol):
    def _with_binding(self, bind: Binding, /) -> Self: ...


class _StateExchange(Protocol[_TD_co]):
    @classmethod
    def _state_default(cls) -> _TD_co:
        """Prepare an "empty" default inner state.

        The vega-lite types use discriminated types a lot, so a type is often defined by
        a required literal/enum.
        """
        ...

    def _unwrap(self) -> _TD_co:
        """Return the inner state."""
        ...

    def _unwrap_clone(self) -> Cloned[_TD_co]:
        """Return a deep-copy of the inner state."""
        return deepcopy(self._unwrap())

    @classmethod
    def _from_common(cls, state: Cloned[_CommonParam], /) -> Self: ...


class _Param(_StateExchange["_CommonParam"]):
    """Experimental fluent parameter builder.

    ## Notes
    - Expose a few arguments at a time
    - Don't let things get into an invalid state
    - For internal use
    """

    def __init__(self, state: _CommonParam, /) -> None:
        self._state: _CommonParam = state

    @classmethod
    def _state_default(cls) -> _CommonParam:
        return {}

    def _unwrap(self) -> _CommonParam:
        return self._state

    @classmethod
    def _from_common(cls, state: Cloned[_CommonParam], /) -> Self:
        return cls(state)

    def interval(
        self, default: Optional[alt.api._SelectionIntervalValueMap] = Undefined
    ) -> _Interval:
        """Create an interval selection parameter, optionally with a default value."""
        state = self._unwrap_clone()
        if not is_undefined(default):
            state["value"] = default
        return _Interval._from_common(state)

    def point(
        self, default: Optional[alt.api._SelectionPointValue] = Undefined
    ) -> _Point:
        """Create a point selection parameter, optionally with a default value."""
        state = self._unwrap_clone()
        if not is_undefined(default):
            state["value"] = default
        return _Point._from_common(state)

    def var(self, default: Any = Undefined) -> _Variable:
        """Create a variable parameter, optionally with a default value."""
        state = self._unwrap_clone()
        if not is_undefined(default):
            state["value"] = default
        return _Variable._from_common(state)

    def expr(
        self, expr: nw.Expr | alt_t.IntoAltExpr, /, default: Any = Undefined
    ) -> _Variable:
        """Create a variable parameter that wraps an expression, optionally with a default value."""
        return self.var(default).expr(expr)

    @property
    def bind(self) -> _BindBuilder:
        return _BindBuilder(self)

    def _with_binding(self, bind: Binding, /) -> Self:
        state = self._unwrap_clone()
        state["bind"] = bind
        return type(self)(state)


class SchemaLike:
    """TODO @dangotbanned: Figure out which classes/states can support this and how."""

    _schema: Final[Mapping[Literal["type"], Literal["object"]]] = {"type": "object"}

    def to_dict(self, *args: Any, **kwds: Any) -> Any: ...


class _Variable(_StateExchange[VariableParamKwds]):
    def __init__(self, state: VariableParamKwds, /) -> None:
        self._state: VariableParamKwds = state

    @classmethod
    def _state_default(cls) -> VariableParamKwds:
        return {}

    def _unwrap(self) -> VariableParamKwds:
        return self._state

    @classmethod
    def _from_common(cls, state: Cloned[_CommonParam], /) -> Self:
        state_next = cls._state_default()
        if state:
            if name := state.pop("name", None):
                state_next["name"] = name
            if bind := state.pop("bind", None):
                state_next["bind"] = bind
            value = state.pop("value", Undefined)
            if not is_undefined(value):
                state_next["value"] = value
            if state:
                raise _invalid_keys_error(state, cls)
        return cls(state_next)

    def expr(self, _: nw.Expr | alt_t.IntoAltExpr, /) -> Self:
        self._state["expr"] = parse_into_vega_expr(_)
        return self

    def react(self, _: Literal[False], /) -> Self:
        self._state["react"] = _
        return self

    @property
    def bind(self) -> _BaseBindBuilder[Self]:
        """Bind the parameter to a UI element/widget."""
        return _BaseBindBuilder(self)

    def _with_binding(self, bind: Binding, /) -> Self:
        state = self._unwrap_clone()
        state["bind"] = bind
        return type(self)(state)


class _Selection(_StateExchange[_SelectionKwdsT]):
    def __init__(self, state: _SelectionKwdsT, /) -> None:
        self._state: _SelectionKwdsT = state

    @classmethod
    def _state_default(cls) -> _SelectionKwdsT:
        raise NotImplementedError

    def _unwrap(self) -> _SelectionKwdsT:
        return self._state

    @classmethod
    def _from_common(cls, state: Cloned[_CommonParam], /) -> Self:
        state_next = cls._state_default()
        if state:
            if name := state.pop("name", None):
                state_next["name"] = name
            if bind := state.pop("bind", None):
                state_next["bind"] = bind
            value = state.pop("value", Undefined)
            if not is_undefined(value):
                state_next["value"] = value
            if state:
                raise _invalid_keys_error(state, cls)
        return cls(state_next)

    def clear(self, _: stream.Stream | stream.StreamSelector | Literal[False], /) -> Self:
        self._state["select"]["clear"] = _
        return self

    def empty(self, _: Literal[False], /) -> Self:
        # NOTE: It isn't a property of a parameter, but can be used
        # when referencing a selection parameter in a predicate
        # https://vega.github.io/vega-lite/docs/parameter.html#as-predicates
        msg = "TODO @dangotbanned: `Parameter.empty` is weird"
        raise NotImplementedError(msg)

    def encodings(self, *encodings: SingleDefUnitChannel_T) -> Self:
        self._state["select"]["encodings"] = encodings
        return self

    def on(self, trigger: stream.Stream | stream.StreamSelector, /) -> Self:
        self._state["select"]["on"] = trigger
        return self

    def resolve(self, _: SelectionResolution_T, /) -> Self:
        self._state["select"]["resolve"] = _
        return self

    def views(self, _: Sequence[str], /) -> Self:
        self._state["views"] = _
        return self

    @property
    def bind(self) -> _BaseBindBuilder[Self]:
        return _BaseBindBuilder(self)

    def _with_binding(self, bind: Binding, /) -> Self:
        state = self._unwrap_clone()
        state["bind"] = bind
        return type(self)(state)


class _Interval(_Selection[_IntervalKwds]):
    @classmethod
    def _state_default(cls) -> _IntervalKwds:
        return {"select": {"type": "interval"}}

    # > Interval selections can only be projected using encodings.
    def mark(self, _: theme.BrushConfigKwds, /) -> Self:
        self._state["select"]["mark"] = _
        return self

    # default is True
    def translate(self, _: stream.StreamSelector | Literal[False], /) -> Self:
        self._state["select"]["translate"] = _
        return self

    # default is True
    def zoom(self, _: stream.StreamSelector | Literal[False], /) -> Self:
        self._state["select"]["zoom"] = _
        return self


class _Point(_Selection[_PointKwds]):
    @classmethod
    def _state_default(cls) -> _PointKwds:
        return {"select": {"type": "point"}}

    def fields(self, *fields: str) -> Self:
        self._state["select"]["fields"] = fields
        return self

    # default is False, only allows bool
    def nearest(self) -> Self:
        self._state["select"]["nearest"] = True
        return self

    # default is True
    def toggle(self, _: nw.Expr | alt_t.IntoAltExpr | Literal[False], /) -> Self:
        self._state["select"]["toggle"] = _ if _ is False else parse_into_vega_expr(_)
        return self


def _param(name: str | None = None) -> _Param:
    return _Param({"name": name} if name else {})


_FromT = TypeVar("_FromT", bound=_WithBinding, covariant=True)  # noqa: PLC0105


class _BaseBindBuilder(Generic[_FromT]):
    _state: _CommonBind
    """Binding state."""
    _param_builder: _FromT
    """The parameter builder we came from."""

    def __init__(self, param_builder: _FromT, /) -> None:
        self._param_builder = param_builder
        self._state = {}

    # NOTE: properties shared by all `Bind*`
    def element(self, css_selector: str, /) -> Self:
        """CSS selector string indicating the parent element to which the input element should be added.

        By default, all input elements are added within the parent container of the Vega view.
        """
        self._state["element"] = css_selector
        return self

    def debounce(self, delay: float, /) -> Self:
        """Delays event handling until `delay` (milliseconds) have elapsed since the last event was fired."""
        self._state["debounce"] = delay
        return self

    def label(self, name: str, /) -> Self:
        """Use `name` as custom label for input elements.

        By default, the signal name is used instead.
        """
        # (not allowed for bind direct)
        self._state["name"] = name
        return self

    # NOTE: The actual bind methods, which return back to where we came from
    def checkbox(self) -> _FromT:
        return self._param_builder._with_binding(
            theme.BindCheckboxKwds(input="checkbox", **self._state)
        )

    def radio(self, options: Sequence[Any], labels: Sequence[str] = ()) -> _FromT:
        # radio buttons
        return self._radio_select("radio", options, labels)

    def select(self, options: Sequence[Any], labels: Sequence[str] = ()) -> _FromT:
        return self._radio_select("select", options, labels)

    dropdown = select

    def range(
        self,
        min: float | None = None,  # noqa: A002
        max: float | None = None,  # noqa: A002
        step: float | None = None,
    ) -> _FromT:
        bind = theme.BindRangeKwds(input="range", **self._state)
        if min is not None:
            bind["min"] = min
        if max is not None:
            bind["max"] = max
        if step is not None:
            bind["step"] = step
        return self._param_builder._with_binding(bind)

    def input(
        self,
        input_type: _ElementType,
        /,
        autocomplete: str | None = None,
        placeholder: str | None = None,
    ) -> _FromT:
        bind = theme.BindInputKwds(input=input_type, **self._state)
        if autocomplete is not None:
            bind["autocomplete"] = autocomplete
        if placeholder is not None:
            bind["placeholder"] = placeholder
        return self._param_builder._with_binding(bind)

    def _radio_select(
        self,
        input_type: Literal["radio", "select"],
        options: Sequence[Any],
        labels: Sequence[str],
        /,
    ) -> _FromT:
        bind = theme.BindRadioSelectKwds(input=input_type, options=options, **self._state)
        if labels:
            bind["labels"] = labels
        return self._param_builder._with_binding(bind)


class _BindBuilder(_BaseBindBuilder[_Param]):
    def scales(
        self, *, encodings: Sequence[SingleDefUnitChannel_T] = ("x", "y")
    ) -> alt.SelectionParameter:
        """Equivalent to `alt.Chart().interactive()`, but without adding the param to the chart."""
        from narwhals._plan.altair.api.parameter import ensure_param_name

        _kwds: Any = self._param_builder._unwrap() | {
            "select": {"type": "interval", "encodings": encodings},
            "bind": "scales",
        }
        return alt.SelectionParameter(**ensure_param_name(_kwds))

    def legend_encoding(
        self, encoding: SingleDefUnitChannel_T = "color", /
    ) -> alt.SelectionParameter:
        from narwhals._plan.altair.api.parameter import ensure_param_name

        _kwds: Any = self._param_builder._unwrap() | {
            "select": {"type": "point", "encodings": [encoding]},
            "bind": "scales",
        }
        return alt.SelectionParameter(**ensure_param_name(_kwds))

    def legend_field(self, field: str, /) -> alt.SelectionParameter:
        from narwhals._plan.altair.api.parameter import ensure_param_name

        _kwds: Any = self._param_builder._unwrap() | {
            "select": {"type": "point", "fields": [field]},
            "bind": "scales",
        }
        return alt.SelectionParameter(**ensure_param_name(_kwds))


def _invalid_keys_error(
    keys: Iterable[str], tp: type[_Variable | _Selection[Any]]
) -> TypeError:
    kind = "variable" if issubclass(tp, _Variable) else "selection"
    msg = f"Keywords {list(keys)!r} cannot be used with {kind!r} parameters"
    return TypeError(msg)


def slider_cutoff() -> _Variable:
    _p1 = _param().var().bind.range(0, 100, 1)
    _p2 = _param().var(50).bind.range(0, 100, 1)
    p3 = _param("slider").var(50).bind.range(0, 100, 1)
    return p3  # noqa: RET504


def slider_point() -> _Point:
    _p1 = (
        _param()
        .bind.label("Release Year")
        .range(1969, 2018, 1)
        .point()
        .fields("Release_Year")
    )
    _p2 = (
        _param("x_select")
        .bind.label("Year ")
        .range(1955, 2005, step=5)
        .point(1980)
        .fields("year")
    )
    return _p2  # noqa: RET504


def misc_params() -> _Variable:
    _projection = (
        _param()
        .var(default="albersUsa")
        .bind.label("Projection ")
        .select(["albers", "albersUsa", "azimuthalEqualArea", "azimuthalEquidistant"])
    )
    _hover_point_opacity = _param().point().on("mouseover").fields("country")
    _search_box = (
        _param().var("").bind.label("Search ").input("search", placeholder="Country")
    )
    return _search_box  # noqa: RET504
