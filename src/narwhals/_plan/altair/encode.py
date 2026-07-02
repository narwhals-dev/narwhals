"""Translate expressions for [`altair.Chart.encode`].

[`altair.Chart.encode`]: https://altair-viz.github.io/user_guide/encodings/index.html
"""

from __future__ import annotations

import datetime as dt
from functools import partial
from typing import TYPE_CHECKING, Final, TypeAlias

from altair import Undefined

from narwhals._plan import expressions as ir
from narwhals._plan.altair import aggregate, typing as alt_t
from narwhals._plan.altair.exceptions import unsupported_error as _unsupported_error
from narwhals._plan.altair.expression import into_vega_expr
from narwhals._plan.expressions import functions as F
from narwhals._plan.expressions.expr import Col, LenStar

if TYPE_CHECKING:
    from typing_extensions import NotRequired, TypedDict
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
    class TestField(_Test, alt_t.Field, TypedDict, closed=True):  # type: ignore[call-arg]
        """When `test`, then `field`."""
    class TestValue(_Test, alt_t.Value, TypedDict, closed=True):  # type: ignore[call-arg]
        """When `test`, then `value`."""
else:
    class TestDatum(_Test, alt_t.Datum, TypedDict):
        """When `test`, then `datum`."""
    class TestField(_Test, alt_t.Field, TypedDict):
        """When `test`, then `field`."""
    class TestValue(_Test, alt_t.Value, TypedDict):
        """When `test`, then `value`."""

class _Conditional(TypedDict):
    condition: TestValue | list[TestValue]

if TYPE_CHECKING:
    class ConditionalDatum(_Conditional, alt_t.Datum, TypedDict, closed=True): ...  # type: ignore[call-arg]
    class ConditionalField(_Conditional, alt_t.Field, TypedDict, closed=True): ...  # type: ignore[call-arg]
    class ConditionalValue(TypedDict, closed=True):  # type: ignore[call-arg]
        condition: TestAny | list[TestValue]
        value: NotRequired[alt_t.InnerValue]
        """Optional else-clause."""
else:
    class ConditionalDatum(_Conditional, alt_t.Datum, TypedDict): ...
    class ConditionalField(_Conditional, alt_t.Field, TypedDict): ...
    class ConditionalValue(_Conditional, alt_t.Value, TypedDict):
        condition: TestAny | list[TestValue]
        value: NotRequired[alt_t.InnerValue]
        """Optional else-clause."""
# fmt: on

TestAny: TypeAlias = TestDatum | TestField | TestValue
"""Any pair of `when(test).then(...)` terms."""


_TP_FIELD = (Col, LenStar, ir.AggExpr, ir.FunctionExpr)
_TP_NATIVE_VALUE = (str, int, bool, float, dt.date, dt.datetime, type(None))

_COND: Final = "condition"
_FIELD: Final = "field"
_TEST: Final = "test"
_VALUE: Final = "value"
_TYPE: Final = "type"
_AGG: Final = "aggregate"
_Q: Final = "quantitative"
unsupported_error = partial(_unsupported_error, target="encoding")


def ternary_expr(expr: ir.TernaryExpr) -> ConditionalValue | ConditionalField:
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
        then_value: TestValue = {_TEST: when, _VALUE: _value(truthy)}
        if isinstance(falsy, ir.Lit):
            return _conditional_value(then_value, falsy)
        if isinstance(falsy, ir.TernaryExpr):
            return _ternary_chained(then_value, falsy)
        return ConditionalField(condition=then_value, **_field(falsy))
    then_field = _field(truthy)
    if isinstance(falsy, ir.Lit):
        return _conditional_value({_TEST: when, **then_field}, falsy)
    if isinstance(falsy, _TP_FIELD):
        msg = f"Only one field may be used within a conditional encoding.\n{falsy!r} followed {truthy!r}, in:\n    {expr!r}"
        raise TypeError(msg)
    if isinstance(falsy, ir.TernaryExpr):
        # NOTE: There's a possibility to do this rewrite here, but I need to move on now
        hint = ""
        if all(not isinstance(e, _TP_FIELD) for e in falsy.iter_left()):
            hint = f"\n\nHint: try rewriting {predicate!r} and moving {truthy!r} to the final `otherwise(...)`"
        msg = f"Chained conditions cannot follow field conditions.\n{falsy!r} followed {truthy!r}, in:\n    {expr!r}{hint}"
        raise TypeError(msg)
    raise unsupported_error(falsy)


def _ternary_chained(
    first: TestValue, otherwise: ir.TernaryExpr
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
    return ConditionalField(condition=conditions, **_field(last))


def _field(expr: ir.ExprIR) -> alt_t.FieldClosed:
    if isinstance(expr, _TP_FIELD):
        match expr:
            case Col(name=name_then):
                return {_FIELD: name_then, _AGG: Undefined, _TYPE: "nominal"}
            case ir.AggExpr():
                return aggregate.from_agg_expr(expr, "encoding")
            case LenStar():
                return {_FIELD: "__count__", _AGG: "count", _TYPE: _Q}
            case ir.FunctionExpr(function=F.NullCount(), args=(Col(name=name),)):
                return {_FIELD: name, _AGG: "missing", _TYPE: _Q}
    raise unsupported_error(expr)


def _value(expr: ir.Lit) -> alt_t.InnerValue:
    v = expr.value
    return v if isinstance(v, _TP_NATIVE_VALUE) else {"expr": into_vega_expr(expr)}


def _conditional_value(
    conditions: TestAny | list[TestValue], otherwise: ir.Lit
) -> ConditionalValue:
    if otherwise.value is None:
        return {_COND: conditions}
    return {_COND: conditions, _VALUE: _value(otherwise)}
