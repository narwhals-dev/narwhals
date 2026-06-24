"""Translation of `TernaryExpr` into a conditional encoding.

So that you can write `nw.when` in the places that `alt.when` currently works.

## Important
Depends on `expression.py`.

## Notes
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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, TypedDict

from narwhals._plan import expressions as ir
from narwhals._plan.altair import typing as alt_t
from narwhals._plan.altair.expression import into_vega_expr
from narwhals._plan.expressions import functions as F


class _Test(TypedDict):
    """A predicate inside a condition.

    Important:
        Only `test`-based predicates can be supported by directly using Narwhals.

        This simplifies the typing for the experiment, but highlights that this can't
        replace everything.
    """

    test: alt_t.Predicate
    """The `when` half of a condition."""


class TestDatum(_Test, alt_t.Datum, TypedDict):
    """When `test`, then `datum`."""


class TestField(_Test, alt_t.Field, TypedDict):
    """When `test`, then `field`."""


if TYPE_CHECKING:

    class TestValue(_Test, alt_t.Value, TypedDict, closed=True):
        """When `test`, then `value`."""
else:

    class TestValue(_Test, alt_t.Value, TypedDict):
        """When `test`, then `value`."""


class _Conditional(TypedDict):
    condition: TestValue | list[TestValue]


class ConditionalDatum(_Conditional, alt_t.Datum, TypedDict): ...


class ConditionalField(_Conditional, alt_t.Field, TypedDict): ...


class ConditionalValue(_Conditional, alt_t.Value, TypedDict):
    condition: TestAny | list[TestValue]


TestAny: TypeAlias = TestDatum | TestField | TestValue
"""Any pair of `when(test).then(...)` terms."""

ConditionalAny: TypeAlias = ConditionalDatum | ConditionalField | ConditionalValue
"""Any fully constructed conditional encoding."""


def ternary(expr: ir.TernaryExpr) -> ConditionalAny:
    """Recursively translate the output of `nw.when(...).then(...).otherwise(...)`."""
    when = into_vega_expr(expr.predicate)
    condition: TestAny
    truthy = expr.truthy
    falsy = expr.falsy
    match truthy:
        case ir.Lit(value=value):
            condition = {"test": when, "value": value}
        case ir.Column(name=name_then):
            condition = {"test": when, "field": name_then, "type": "nominal"}

        # TODO @dangotbanned: Some parallel to `parse_shorthand`
        # `falsy` has made a start
        case _:
            msg = f"Unsupported {truthy!r}"
            raise NotImplementedError(msg)
    match falsy:
        # TODO @dangotbanned: `When` (chained)
        # TODO @dangotbanned: here's where we create a list
        case ir.TernaryExpr():
            msg = f"todo (chained ternary) {type(falsy).__name__!r}"
            raise NotImplementedError(msg)

        case ir.Lit(value=value):
            return {"condition": condition, "value": value}
        case ir.Column(name=name_otherwise):
            if "field" in condition:
                msg_0 = (
                    "Only one field may be used within a conditional encoding.\n"
                    f"{falsy!r} followed {truthy!r}, in:\n    {expr!r}"
                )
                raise TypeError(msg_0)
            return {"condition": condition, "field": name_otherwise, "type": "nominal"}

        # TODO @dangotbanned: Aggregate
        # NOTE: `narwhals._plan.altair.aggregate.AGG_EXPR*`
        case ir.AggExpr() | ir.Len():
            msg = f"todo (aggregate) {type(falsy).__name__!r}"
            raise NotImplementedError(msg)

        # NOTE: `narwhals._plan.altair.aggregate.AGG_FUNC`
        case ir.FunctionExpr(function=F.NullCount(), args=(ir.Column(name=name),)):
            if "field" in condition:
                msg_0 = (
                    "Only one field may be used within a conditional encoding.\n"
                    f"{falsy.args[0]!r} followed {truthy!r}, in:\n    {expr!r}"
                )
                raise TypeError(msg_0)
            return {
                "condition": condition,
                "field": name,
                "aggregate": "missing",
                "type": "quantitative",
            }
        case _:
            msg = f"todo {type(falsy).__name__!r}"
            raise NotImplementedError(msg)
