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

import functools
import hashlib
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypedDict

import altair as alt
from altair import theme

from narwhals._plan.altair.expression import parse_into_vega_expr

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import altair.vegalite.v6.schema._typing as _alt_t
    from _typeshed import Incomplete
    from typing_extensions import TypedDict, Unpack

    import narwhals._plan as nw
    from narwhals._plan.altair import typing as alt_t

    class _Name(TypedDict, total=False, closed=True):  # type: ignore[call-arg]
        name: str


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


SelectionConfig: TypeAlias = (
    theme.IntervalSelectionConfigKwds | theme.PointSelectionConfigKwds
)
"""Any config accepted by `SelectionParam(select=...)`.

Note:
    They also allow `Literal['point', 'interval']`.
"""

ParamKwds: TypeAlias = theme.VariableParameterKwds | theme.TopLevelSelectionParameterKwds
"""Pre-construction state for a parameter.

No arguments are required yet.
"""


def variable(
    expr: nw.Expr | alt_t.IntoAltExpr, /, **kwds: Unpack[theme.VariableParameterKwds]
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
    """

    # NOTE: These 3 don't follow the pattern I was thinking of here, since they work in one step
    @staticmethod
    def bind_scales(
        *,
        encodings: Sequence[_alt_t.SingleDefUnitChannel_T] = ("x", "y"),
        **kwds: Unpack[_Name],
    ) -> SelectionParam:
        """Equivalent to `alt.Chart().interactive()`, but without adding the param to the chart."""
        return _selection(
            **kwds, select=_config_interval(encodings=encodings), bind="scales"
        )

    # NOTE: 1 encoding or 1 field -> one method each
    # https://github.com/vega/altair/blob/48b388f140c79d29056d6ea56e519b27e2ed8838/tests/examples_methods_syntax/interactive_legend.py#L5-L6
    @staticmethod
    def bind_legend_encoding(
        encoding: _alt_t.SingleDefUnitChannel_T = "color", /, **kwds: Unpack[_Name]
    ) -> SelectionParam:
        return _selection(
            **kwds, select=_config_point(encodings=[encoding]), bind="legend"
        )

    @staticmethod
    def bind_legend_field(field: str, /, **kwds: Unpack[_Name]) -> SelectionParam:
        return _selection(**kwds, select=_config_point(fields=[field]), bind="legend")


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


def _config_interval(
    **kwds: Unpack[theme.IntervalSelectionConfigKwds],
) -> theme.IntervalSelectionConfigKwds:
    # discriminator field is the only required argument
    kwds["type"] = "interval"
    return kwds


def _config_point(
    **kwds: Unpack[theme.PointSelectionConfigKwds],
) -> theme.PointSelectionConfigKwds:
    # discriminator field is the only required argument
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
