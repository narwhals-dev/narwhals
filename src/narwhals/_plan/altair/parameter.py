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
import functools
import hashlib
from copy import deepcopy
from importlib.util import find_spec
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeVar,
    final,
)

import altair as alt
from altair import Undefined, theme
from altair.theme import VariableParameterKwds as VariableParamKwds
from altair.utils import is_undefined

from narwhals._plan.altair.expression import parse_into_vega_expr

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import altair.vegalite.v6.schema._typing as _alt_t
    import typing_extensions as te
    from _typeshed import Incomplete
    from typing_extensions import Self, Unpack

    import narwhals._plan as nw
    from narwhals._plan.altair import stream, typing as alt_t

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

BindingSelect: TypeAlias = (
    Binding | theme.LegendStreamBindingKwds | Literal["legend", "scales"]
)
"""Anything accepted by `selection_*(bind=...)`.

Important:
    Wider than `VariableParameter`.
"""

VariableParam: TypeAlias = alt.VariableParameter
"""A parameter that wraps an expression."""

SelectionParam: TypeAlias = alt.SelectionParameter | alt.TopLevelSelectionParameter
"""A parameter that [defines a data query] driven by user input (e.g., mouse clicks or drags).

Accepts config via a `select` argument.

[defines a data query]: https://vega.github.io/vega-lite/docs/selection.html
"""

IntervalConfig: TypeAlias = theme.IntervalSelectionConfigKwds
PointConfig: TypeAlias = theme.PointSelectionConfigKwds
SelectionConfig: TypeAlias = IntervalConfig | PointConfig
"""Any config accepted by `SelectionParam(select=...)`.

Important:
    Avoid using these `TypedDict`s directly, as the corresponding classes have a required (at validation-time)
    discriminator field `type`. But that isn't captured in the `TypedDict` or `SchemaBase` definitions.
"""


ParamKwds: TypeAlias = VariableParamKwds | theme.TopLevelSelectionParameterKwds
"""Pre-construction state for a parameter.

No arguments are required yet.
"""

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


class _WithBind(Protocol):
    """Can't think of a good name yet."""

    def _with_bind(self, bind: Binding, /) -> Self: ...


_PBuild = TypeVar("_PBuild", bound=_WithBind, covariant=True)  # noqa: PLC0105


def variable(
    expr: nw.Expr | alt_t.IntoAltExpr, /, **kwds: Unpack[VariableParamKwds]
) -> VariableParam:
    """Create a named variable parameter."""
    kwds["expr"] = parse_into_vega_expr(expr)
    kwds["name"] = _param_name(kwds)
    return alt.VariableParameter(**kwds)


def _selection(**kwds: Unpack[theme.TopLevelSelectionParameterKwds]) -> SelectionParam:
    """Required: select."""
    kwds["name"] = _param_name(kwds)
    if "views" in kwds:
        return alt.TopLevelSelectionParameter(**kwds)
    # this one doesn't have a typed dict, but the only difference vs the other is `views`
    return alt.SelectionParameter(**kwds)


class _ParamBuilder:
    """Experimental fluent parameter builder.

    ## Notes
    - Expose a few arguments at a time
    - Don't let things get into an invalid state
    - For internal use
    """

    _state: _CommonParam

    def __init__(self, state: _CommonParam, /) -> None:
        self._state = state

    @property
    def bind(self) -> _BindBuilder[Self]:
        return _BindBuilder(self)

    def _with_bind(self, bind: Binding, /) -> Self:
        if self._state.get("bind"):
            msg = "Keyword 'bind' was provided multiple times"
            raise TypeError(msg)
        next_state = deepcopy(self._state)
        next_state["bind"] = bind
        return type(self)(next_state)

    def point(self, default: Any = Undefined) -> _PointBuilder:
        msg = "TODO selection_point"
        raise NotImplementedError(msg)

    def interval(self) -> _IntervalBuilder:
        msg = "TODO selection_interval"
        raise NotImplementedError(msg)

    # NOTE: variable parameters do not require an expression
    # most examples just add a default + binding
    def expr(
        self, expr: nw.Expr | alt_t.IntoAltExpr, *, react: bool = True
    ) -> _VariableParamBuilder:
        return _VariableParamBuilder._from_param_expr(self, expr, react=react)

    def var(self, default: Any = Undefined) -> _VariableParamBuilder:
        """Create a variable parameter, optionally with a default value."""
        return _VariableParamBuilder._from_param_var(self, default)


class _BaseSelectionBuilder:
    def clear(
        self, clear: stream.Stream | stream.StreamSelector | Literal[False], /
    ) -> Self: ...
    def encodings(self, *encodings: _alt_t.SingleDefUnitChannel_T) -> Self: ...
    def on(self, trigger: stream.Stream | stream.StreamSelector, /) -> Self: ...
    def resolve(self, resolve: _alt_t.SelectionResolution_T, /) -> Self: ...


class _IntervalBuilder(_BaseSelectionBuilder):
    # > Interval selections can only be projected using encodings.
    def mark(self, mark: theme.BrushConfigKwds, /) -> Self: ...
    # default is True
    def translate(self, translate: stream.StreamSelector | Literal[False], /) -> Self: ...
    # default is True
    def zoom(self, zoom: stream.StreamSelector | Literal[False], /) -> Self: ...


class _PointBuilder(_BaseSelectionBuilder):
    def fields(self, *fields: str) -> Self: ...
    # default is False, only allows bool
    def nearest(self) -> Self: ...
    # default is True
    def toggle(self, toggle: nw.Expr | alt_t.IntoAltExpr | Literal[False], /) -> Self: ...


@final
class _VariableParamBuilder:
    _state: VariableParamKwds

    def __init__(self, state: VariableParamKwds, /) -> None:
        self._state = state

    @property
    def bind(self) -> _BaseBindBuilder[Self]:
        """Bind the parameter to a UI element/widget."""
        return _BaseBindBuilder(self)

    def _with_bind(self, bind: Binding, /) -> Self:
        if self._state.get("bind"):
            msg = "Keyword 'bind' was provided multiple times"
            raise TypeError(msg)
        next_state = deepcopy(self._state)
        next_state["bind"] = bind
        return type(self)(next_state)

    @staticmethod
    def _from_param_var(builder: _ParamBuilder, default: Any) -> _VariableParamBuilder:
        prev_state = deepcopy(builder._state)
        next_state: VariableParamKwds = {}
        if prev_state:
            if name := prev_state.pop("name", None):
                next_state["name"] = name
            if bind := prev_state.pop("bind", None):
                next_state["bind"] = bind
            value = prev_state.pop("value", Undefined)
            if not is_undefined(value):
                if not is_undefined(default):
                    msg = "Keyword 'value'/'default' was provided multiple times"
                    raise TypeError(msg)
                next_state["value"] = value

            if prev_state:
                msg = (
                    f"Keywords {list(prev_state)!r} are not valid for variable parameters"
                )
                raise TypeError(msg)
        elif not is_undefined(default):
            next_state["value"] = default

        return _VariableParamBuilder(next_state)

    @staticmethod
    def _from_param_expr(
        builder: _ParamBuilder, expr: nw.Expr | alt_t.IntoAltExpr, *, react: bool
    ) -> _VariableParamBuilder:
        prev_state = deepcopy(builder._state)
        next_state: VariableParamKwds = {}
        if prev_state:
            if name := prev_state.pop("name", None):
                next_state["name"] = name
            if bind := prev_state.pop("bind", None):
                next_state["bind"] = bind
            value = prev_state.pop("value", Undefined)
            if not is_undefined(value):
                next_state["value"] = value

            if prev_state:
                msg = (
                    f"Keywords {list(prev_state)!r} are not valid for variable parameters"
                )
                raise TypeError(msg)
        next_state.update({"expr": parse_into_vega_expr(expr), "react": react})
        return _VariableParamBuilder(next_state)

    def default(self, value: Any, /) -> _VariableParamBuilder:
        self._state["value"] = value
        return self


def _param(name: str | None = None) -> _ParamBuilder:
    return _ParamBuilder({"name": name} if name else {})


class _BaseBindBuilder(Generic[_PBuild]):
    _state: _CommonBind
    """Binding state."""
    _param_builder: _PBuild
    """The parameter builder we came from."""

    def __init__(self, param_builder: _PBuild, /) -> None:
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
    def checkbox(self) -> _PBuild:
        return self._param_builder._with_bind(
            theme.BindCheckboxKwds(input="checkbox", **self._state)
        )

    def radio(self, options: Sequence[Any], labels: Sequence[str] = ()) -> _PBuild:
        # radio buttons
        return self._radio_select("radio", options, labels)

    def select(self, options: Sequence[Any], labels: Sequence[str] = ()) -> _PBuild:
        return self._radio_select("select", options, labels)

    dropdown = select

    def range(
        self,
        min: float | None = None,  # noqa: A002
        max: float | None = None,  # noqa: A002
        step: float | None = None,
    ) -> _PBuild:
        bind = theme.BindRangeKwds(input="range", **self._state)
        if min is not None:
            bind["min"] = min
        if max is not None:
            bind["max"] = max
        if step is not None:
            bind["step"] = step
        return self._param_builder._with_bind(bind)

    def input(
        self,
        input_type: _ElementType,
        /,
        autocomplete: str | None = None,
        placeholder: str | None = None,
    ) -> _PBuild:
        bind = theme.BindInputKwds(input=input_type, **self._state)
        if autocomplete is not None:
            bind["autocomplete"] = autocomplete
        if placeholder is not None:
            bind["placeholder"] = placeholder
        return self._param_builder._with_bind(bind)

    def _radio_select(
        self,
        input_type: Literal["radio", "select"],
        options: Sequence[Any],
        labels: Sequence[str],
        /,
    ) -> _PBuild:
        bind = theme.BindRadioSelectKwds(input=input_type, options=options, **self._state)
        if labels:
            bind["labels"] = labels
        return self._param_builder._with_bind(bind)


# TODO @dangotbanned: Fix `self._param_builder._state` dependency
class _BindBuilder(_BaseBindBuilder[_PBuild]):
    def scales(
        self, *, encodings: Sequence[_alt_t.SingleDefUnitChannel_T] = ("x", "y")
    ) -> SelectionParam:
        """Equivalent to `alt.Chart().interactive()`, but without adding the param to the chart."""
        # NOTE: Obvious now that `_selection` is the wrong API, when combined with this
        select = _config_interval(encodings=encodings)
        if name := self._param_builder._state.get("name"):  # pyright: ignore[reportAttributeAccessIssue]
            return _selection(bind="scales", select=select, name=name)
        return _selection(bind="scales", select=select)

    # NOTE: 1 encoding or 1 field -> one method each
    # https://github.com/vega/altair/blob/48b388f140c79d29056d6ea56e519b27e2ed8838/tests/examples_methods_syntax/interactive_legend.py#L5-L6

    def legend_encoding(
        self, encoding: _alt_t.SingleDefUnitChannel_T = "color", /
    ) -> SelectionParam:
        select = _config_point(encodings=[encoding])
        if name := self._param_builder._state.get("name"):  # pyright: ignore[reportAttributeAccessIssue]
            return _selection(bind="legend", select=select, name=name)
        return _selection(bind="legend", select=select)

    def legend_field(self, field: str, /) -> SelectionParam:
        select = _config_point(fields=[field])
        if name := self._param_builder._state.get("name"):  # pyright: ignore[reportAttributeAccessIssue]
            return _selection(bind="legend", select=select, name=name)
        return _selection(bind="legend", select=select)


def slider_cutoff() -> _VariableParamBuilder:
    _p1 = _param().var().bind.range(0, 100, 1)
    _p2 = _param().var(50).bind.range(0, 100, 1)
    p3 = _param("slider").var(50).bind.range(0, 100, 1)
    return p3  # noqa: RET504


def slider_point() -> _PointBuilder:
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


def _variable(
    expr: nw.Expr | alt_t.IntoAltExpr,
    *,
    name: str | None,
    bind: Binding,
    value: Any,
    react: bool,
) -> Incomplete:
    raise NotImplementedError


# too many parameters!
def _selection_interval(
    *,
    # common
    name: str | None,
    bind: BindingSelect,
    value: alt.api._SelectionIntervalValueMap,
    views: Sequence[str],
    # interval_config (`select` expands to)
    # # common
    clear: str | bool | theme.MergedStreamKwds | theme.DerivedStreamKwds,
    encodings: Sequence[_alt_t.SingleDefUnitChannel_T],
    fields: Sequence[str],
    on: str | theme.MergedStreamKwds | theme.DerivedStreamKwds,
    resolve: _alt_t.SelectionResolution_T,
    # # interval
    mark: theme.BrushConfigKwds,
    translate: str | bool,
    zoom: str | bool,
) -> Incomplete:
    """This & selection_point have too many parameters!"""
    raise NotImplementedError


# too many parameters!
def _selection_point(
    *,
    # common
    name: str | None,
    bind: BindingSelect,
    value: alt.api._SelectionPointValue,
    views: Sequence[str],
    # point_config (`select` expands to)
    # # common
    clear: str | bool | theme.MergedStreamKwds | theme.DerivedStreamKwds,
    encodings: Sequence[_alt_t.SingleDefUnitChannel_T],
    fields: Sequence[str],
    on: str | theme.MergedStreamKwds | theme.DerivedStreamKwds,
    resolve: _alt_t.SelectionResolution_T,
    # # point
    nearest: bool,
    toggle: nw.Expr | alt_t.IntoAltExpr | bool,  # the other strings are all unrelated
) -> Incomplete:
    raise NotImplementedError


def _param_name(kwds: ParamKwds, /) -> str:
    if name := kwds.pop("name", None):
        return name
    encoded = serialize(kwds, deterministic=True)
    # NOTE: https://github.com/vega/altair/pull/3291#issuecomment-1866999185
    # - 256 vs 224 -> 64 vs 56 characters (only need 16)
    # - not used for security
    return f"param_{hashlib.sha224(encoded, usedforsecurity=False).hexdigest()[:16]}"


def _config_interval(**kwds: Unpack[IntervalConfig]) -> IntervalConfig:
    kwds["type"] = "interval"
    return kwds


def _config_point(**kwds: Unpack[PointConfig]) -> PointConfig:
    kwds["type"] = "point"
    return kwds


def serialize(
    obj: Any, /, *, deterministic: bool, default: Callable[[Any], Any] | None = str
) -> bytes:
    """Serialize an object to bytes.

    Arguments:
        obj: An object composed of JSON compatible types.
        deterministic: Ensure the same input produces the same output bytes.
            Use this when serializing to generate a hash value.

            Does not guarantee anything beyond the description in [`msgspec.json.encode(order="deterministic")`]

        default: A hook for objects that can't otherwise be serialized.
            The caller is responsible for providing a hook that roundtrips,
            which `default=str` cannot.

    Important:
        Uses the fastest available json serializer, in priority of [`msgspec`] > [`orjson`] > [`json`].

    [`msgspec`]: https://github.com/msgspec/msgspec
    [`orjson`]: https://github.com/ijl/orjson
    [`json`]: https://docs.python.org/3/library/json.html
    [`msgspec.json.encode(order="deterministic")`]: https://msgspec.dev/api#msgspec.json.encode
    """
    return _build_serializer(deterministic=deterministic, default=default)(obj)


@functools.cache
def _build_serializer(
    *, deterministic: bool = False, default: Callable[[Any], Any] | None = str
) -> Callable[[Any], bytes]:
    """Return the fastest available json serializer."""
    # NOTE: `marimo` depends on `msgspec`, so take it if available
    if find_spec("msgspec"):
        from msgspec.json import Encoder

        return Encoder(
            order="deterministic" if deterministic else None, enc_hook=default
        ).encode

    # NOTE: `mypy`, `jupyter_client` optionally depend on `orjson`
    if find_spec("orjson"):
        import orjson  # type: ignore[import-not-found]

        orjson_encode: Callable[[Any], bytes] = functools.partial(
            orjson.dumps,
            default=default,
            option=orjson.OPT_SORT_KEYS if deterministic else None,
        )
        return orjson_encode
    from json import JSONEncoder

    _encode = JSONEncoder(
        sort_keys=deterministic, default=default, separators=(",", ":")
    ).encode

    def encode(obj: Any, /) -> bytes:
        return _encode(obj).encode()

    return encode
