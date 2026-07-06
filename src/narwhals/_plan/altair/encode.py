"""Translate expressions for [`altair.Chart.encode`].

[`altair.Chart.encode`]: https://altair-viz.github.io/user_guide/encodings/index.html
"""

from __future__ import annotations

import datetime as dt
import functools
from typing import TYPE_CHECKING, Final, TypeAlias, overload

from altair import Undefined

import narwhals.stable.v1 as stable_v1
from narwhals._plan import expressions as ir
from narwhals._plan.altair import aggregate, typing as alt_t
from narwhals._plan.altair.exceptions import unsupported_error as _unsupported_error
from narwhals._plan.altair.expression import into_vega_expr
from narwhals._plan.altair.typing import Channel, Field, Value
from narwhals._plan.expressions import functions as F
from narwhals._plan.expressions.expr import Col, LenStar

if TYPE_CHECKING:
    from typing_extensions import NotRequired, TypedDict

    from narwhals._plan.altair.chart import Chart as NwChart
    from narwhals._plan.altair.typing import Optional, VegaType
    from narwhals.dtypes import DType
else:
    from typing import TypedDict


class _Test(TypedDict):
    """A predicate inside a condition.

    Important:
        Only `test`-based predicates can be supported by directly using Narwhals.

        This simplifies the typing for the experiment, but highlights that this can't
        replace everything.
    """

    test: alt_t.Predicate
    """The `when` half of a condition."""


# fmt: off
if TYPE_CHECKING:
    # https://github.com/python/mypy/pull/21382
    class TestDatum(_Test, alt_t.Datum, TypedDict, closed=True):  # type: ignore[call-arg]
        """When `test`, then `datum`."""
    class TestField(_Test, alt_t.FieldOpen, TypedDict, closed=True):  # type: ignore[call-arg]
        """When `test`, then `field`."""
    class TestValue(_Test, alt_t.ValueOpen, TypedDict, closed=True):  # type: ignore[call-arg]
        """When `test`, then `value`."""
else:
    class TestDatum(_Test, alt_t.Datum, TypedDict):
        """When `test`, then `datum`."""
    class TestField(_Test, alt_t.FieldOpen, TypedDict):
        """When `test`, then `field`."""
    class TestValue(_Test, alt_t.ValueOpen, TypedDict):
        """When `test`, then `value`."""

class _Conditional(TypedDict):
    condition: TestValue | list[TestValue]

if TYPE_CHECKING:
    class ConditionalDatum(_Conditional, alt_t.Datum, TypedDict, closed=True): ...  # type: ignore[call-arg]
    class ConditionalField(_Conditional, alt_t.FieldOpen, TypedDict, closed=True): ...  # type: ignore[call-arg]
    class ConditionalValue(TypedDict, closed=True):  # type: ignore[call-arg]
        condition: TestAny | list[TestValue]
        value: NotRequired[alt_t.InnerValue]
        """Optional else-clause."""
else:
    class ConditionalDatum(_Conditional, alt_t.Datum, TypedDict): ...
    class ConditionalField(_Conditional, alt_t.FieldOpen, TypedDict): ...
    class ConditionalValue(_Conditional, alt_t.ValueOpen, TypedDict):
        condition: TestAny | list[TestValue]
        value: NotRequired[alt_t.InnerValue]
        """Optional else-clause."""
# fmt: on

TestAny: TypeAlias = TestDatum | TestField | TestValue
"""Any pair of `when(test).then(...)` terms."""


Encoded: TypeAlias = ConditionalValue | ConditionalField | Field | Value
"""Anything that has been translated for `altair.Chart.encode`."""

_TP_FIELD = (Col, LenStar, ir.AggExpr, ir.FunctionExpr)
_TP_NATIVE_VALUE = (str, int, bool, float, dt.date, dt.datetime, type(None))

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

_COND: Final = "condition"
_FIELD: Final = "field"
_TEST: Final = "test"
_VALUE: Final = "value"
_TYPE: Final = "type"
_AGG: Final = "aggregate"
_Q: Final = "quantitative"
unsupported_error = functools.partial(_unsupported_error, target="encoding")


@overload
def from_expr(expr: ir.Lit, chart: NwChart, channel: Channel | None = None) -> Value: ...
@overload
def from_expr(
    expr: ir.TernaryExpr, chart: NwChart, channel: Channel | None = None
) -> ConditionalValue | ConditionalField: ...
@overload
def from_expr(
    expr: Col | LenStar | ir.AggExpr | ir.FunctionExpr,
    chart: NwChart,
    channel: Channel | None = None,
) -> Field: ...
@overload
def from_expr(
    expr: ir.ExprIR, chart: NwChart, channel: Channel | None = None
) -> Encoded: ...
def from_expr(expr: ir.ExprIR, chart: NwChart, channel: Channel | None = None) -> Encoded:
    result = _from_expr(expr, chart, channel)
    if result is None:
        raise unsupported_error(expr)
    return result


@functools.singledispatch
def _from_expr(
    expr: ir.ExprIR,  # noqa: ARG001
    chart: NwChart,  # noqa: ARG001
    channel: Channel | None,  # noqa: ARG001
    /,
) -> Encoded | None:
    return None


@_from_expr.register(ir.TernaryExpr)
def ternary_expr(
    expr: ir.TernaryExpr, chart: NwChart, channel: Channel | None, /
) -> ConditionalValue | ConditionalField:
    """Translation of `TernaryExpr` into a conditional encoding.

    So that you can write `nw.when` in the places that `alt.when` currently works.

    ## Notes (1)
    `expression.py` handles `TernaryExpr` (including those that are nested) by mapping them directly to `if_`:

        expr: ir.TernaryExpr
        alt.expr.if_(expr.predicate, expr.truthy, expr.falsy)
        # expr.predicate ? expr.truthy : expr.falsy

    - That allows using `nw.when` in places that `alt.when` is **not yet supported**
        - because the IR handles the chaining of `when().then().when()...` independent of `altair`.
    - `alt.when` [internally constructs a `dict`] in the same shape as a [`Condition`]
        - which cannot be passed to places that only accept a `str`.

    [`Condition`]: https://vega.github.io/vega-lite/docs/condition.html
    [internally constructs a `dict`]: https://github.com/vega/altair/blob/bc3c353c21dbc7b43a110b48f887facebaf7772f/altair/vegalite/v6/api.py#L575-L1391

    ## Notes (2)
    [Vega-Lite docs]: https://vega.github.io/vega-lite/docs/condition.html

    I found the [Vega-Lite docs] hard to follow. In short, they mean:

    - A field can never be in a list of conditions
    - The final (optional) otherwise is where we determine if it is a field
        - It is a value by default
    - we can do any of these:

        condition=(test=..., value=...)                   # value
        condition=(test=..., value=...), field=...        # field
        condition=[(test=..., value=...)]                 # value
        condition=[(test=..., value=...)], field=...      # field

        condition=(test=..., field=..., ...)              # value
        condition=(test=..., field=..., ...), value=...   # value
    """
    predicate, truthy, falsy = expr.predicate, expr.truthy, expr.falsy
    when = into_vega_expr(predicate)

    if isinstance(truthy, ir.Lit):
        then_value: TestValue = {_TEST: when, **from_expr(truthy, chart, None)}
        if isinstance(falsy, ir.Lit):
            return _conditional_value(then_value, falsy)
        if isinstance(falsy, ir.TernaryExpr):
            return _ternary_chained(then_value, falsy, chart, channel)
        return ConditionalField(condition=then_value, **_field(falsy, chart, channel))
    then_field = _field(truthy, chart, channel)
    if isinstance(falsy, ir.Lit):
        return _conditional_value({_TEST: when, **then_field}, falsy)
    if isinstance(falsy, _TP_FIELD):
        msg = f"Only one field may be used within a conditional encoding.\n{falsy!r} followed {truthy!r}, in:\n    {expr!r}"
        raise TypeError(msg)
    if isinstance(falsy, ir.TernaryExpr):
        # NOTE: There's a possibility to do this rewrite here, but I need to move on now
        msg = f"Chained conditions cannot follow field conditions.\n{falsy!r} followed {truthy!r}, in:\n    {expr!r}"
        raise TypeError(msg)
    raise unsupported_error(falsy)


def _ternary_chained(
    first: TestValue, otherwise: ir.TernaryExpr, chart: NwChart, channel: Channel | None
) -> ConditionalValue | ConditionalField:
    conditions = [first]
    last: ir.TernaryExpr | ir.ExprIR = otherwise
    while isinstance(last, ir.TernaryExpr):
        truthy = last.truthy
        if not isinstance(truthy, ir.Lit):
            if isinstance(truthy, _TP_FIELD):
                msg = f"Chained conditions cannot contain field conditions.\nFound {truthy!r}, in:\n    {otherwise!r}"
                raise TypeError(msg)
            raise unsupported_error(truthy)
        conditions.append({_TEST: into_vega_expr(last.predicate), _VALUE: _value(truthy)})
        last = last.falsy
    if isinstance(last, ir.Lit):
        return _conditional_value(conditions, last)
    return ConditionalField(condition=conditions, **_field(last, chart, channel))


@_from_expr.register(Col)
def _col(expr: Col, chart: NwChart, channel: Channel | None, /) -> Field:
    field = expr.name
    if vtype := vegalite_type(field, chart, channel):
        return {_FIELD: field, _AGG: Undefined, _TYPE: vtype}
    return {_FIELD: field, _AGG: Undefined}


@_from_expr.register(ir.AggExpr)
def _agg_expr(expr: ir.AggExpr, chart: NwChart, channel: Channel | None, /) -> Field:
    encoding = aggregate.from_agg_expr(expr, context="encoding")
    if vtype := vegalite_type(encoding[_FIELD], chart, channel, _Q):
        encoding[_TYPE] = vtype
    return encoding


@_from_expr.register(LenStar)
def _len(_: LenStar, __: NwChart, channel: Channel | None, /) -> Field:
    if channel and channel in _SECONDARY_FIELD:
        return {_FIELD: "__count__", _AGG: "count"}
    return {_FIELD: "__count__", _AGG: "count", _TYPE: _Q}


@_from_expr.register(ir.FunctionExpr)
def _function_expr(
    expr: ir.FunctionExpr, _: NwChart, channel: Channel | None, /
) -> Field | None:
    match expr:
        case ir.FunctionExpr(function=func, args=(Col(name=name),)):
            field: Field = {_FIELD: name, _AGG: Undefined}
            if type(func) is F.NullCount:
                field[_AGG] = "missing"
            elif isinstance(func, F.HistBinCount):
                field["bin"] = {"maxbins": func.bin_count}
            elif isinstance(func, F.HistBins):
                field["bin"] = {"steps": func.bins}
            else:
                return None
            if channel and channel in _SECONDARY_FIELD:
                return field
            field[_TYPE] = _Q
            return field
        case _:
            return None


@_from_expr.register(ir.Lit)
def _lit(expr: ir.Lit, _: NwChart, __: Channel | None, /) -> Value:
    # NOTE: https://altair-viz.github.io/user_guide/encodings/index.html#datum-and-value
    # A heuristic would probably be too complex:
    # - via the context of `channel` & `expr.dtype`
    # - use datum when there's a scale?
    #   - Very rarely seen `alt.datum(...)` in use
    return {"value": _value(expr)}


def _field(expr: ir.ExprIR, chart: NwChart, channel: Channel | None) -> Field:
    if not isinstance(expr, _TP_FIELD):
        raise unsupported_error(expr)
    return from_expr(expr, chart, channel)


def _value(expr: ir.Lit) -> alt_t.InnerValue:
    v = expr.value
    return v if isinstance(v, _TP_NATIVE_VALUE) else {"expr": into_vega_expr(expr)}


def _conditional_value(
    conditions: TestAny | list[TestValue], otherwise: ir.Lit
) -> ConditionalValue:
    if otherwise.value is None:
        return {_COND: conditions}
    return {_COND: conditions, _VALUE: _value(otherwise)}


def vegalite_type(
    field: str,
    chart: NwChart,
    channel: Channel | None,
    /,
    default: Optional[VegaType] = Undefined,
) -> Optional[VegaType] | None:
    if channel and channel in _SECONDARY_FIELD:
        return None
    return (
        _vegalite_type(dtype)
        if (dtype := chart._try_collect_schema.get(field))
        else default
    )


@functools.lru_cache(16)
def _vegalite_type(dtype: DType, /) -> Optional[VegaType]:
    if dtype.is_numeric():
        return "quantitative"
    if isinstance(dtype, (stable_v1.String, stable_v1.Categorical, stable_v1.Boolean)):
        return "nominal"
    if isinstance(dtype, (stable_v1.Datetime, stable_v1.Date)):
        return "temporal"
    return Undefined
