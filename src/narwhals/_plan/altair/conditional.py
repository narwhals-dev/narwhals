"""Translation of `TernaryExpr` into a conditional encoding.

So that you can write `nw.when` in the places that `alt.when` currently works.

## Important
Depends on both `expression.py` and `predicate.py`.

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
