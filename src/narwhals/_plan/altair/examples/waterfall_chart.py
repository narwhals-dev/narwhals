"""[Waterfall Chart].

[Waterfall Chart]: https://altair-viz.github.io/gallery/waterfall_chart.html

The original was an impressive feat, but required combining 6 different ways to write an expression in Altair:

1. `alt.datum.<column-name>`
2. `alt.expr.<expression-method-name>(<into-vega-expr-string>, ...)`
3. Operator overloading on the above
4. `.transform_window(<new-column-name> = "<aggregate-or-window-op-name>(<column-name>)")`
5. `.transform_calculate(<new-column-name> = <vega-expr-string> | alt.Expr)`
6. `alt.when(<most-of-the-above> | alt.Parameter).then(<shorthand-string> | alt.value | dict | alt.SchemaBase).when(...).then(...)...`

The above is not even every *possible* way to write something that looks like an expression 😵

---

Knowing when you can use each - or even what they all are - is non-trivial:

- `alt.datum.<column-name>` stops working when the column has a space
    - you then need to use `alt.datum["<column-name>"]`
- Magic strings cannot be type checked and appear in many places
    - `"lead(label)"` is valid in one place
    - `"sum(amount)"` is valid in many places, but is not a `<vega-expr-string>`
- Operator overloading often degrades to `Unknown`
- Most signatures say they accept `dict | alt.SchemaBase | str`
    - but have more complicated runtime rules
    - nothing is accepted everywhere

Note:
    Writing a Narwhals expression produces an `ExprIR` that can translate to every use case in this chart.

    It is also *immutable-by-design*, so the translation can be cached on any hot paths.
"""

from __future__ import annotations

import polars as pl

import narwhals._plan as nw
from narwhals._plan.altair import api

data = [
    {"label": "Begin", "amount": 4000},
    {"label": "Jan", "amount": 1707},
    {"label": "Feb", "amount": -1425},
    {"label": "Mar", "amount": -1030},
    {"label": "Apr", "amount": 1812},
    {"label": "May", "amount": -1067},
    {"label": "Jun", "amount": -1481},
    {"label": "Jul", "amount": 1228},
    {"label": "Aug", "amount": 1176},
    {"label": "Sep", "amount": 1146},
    {"label": "Oct", "amount": 1205},
    {"label": "Nov", "amount": -1388},
    {"label": "Dec", "amount": 1492},
    {"label": "End", "amount": 0},
]
source = pl.DataFrame(data)

# Define frequently referenced fields
amount = nw.col("amount")
label = nw.col("label")

# NOTE: Both are defined in `transform_window`, but referenced as names?
window_lead_label = nw.col("window_lead_label")
window_sum_amount = nw.col("window_sum_amount")

# New stuff
when_end = nw.when(label=nw.lit("End"))
begin_or_end = label.is_in(["Begin", "End"])


calc_prev_sum = when_end.then(0).otherwise(window_sum_amount - amount)
calc_amount = when_end.then(window_sum_amount).otherwise(amount)


base = (
    api.Chart(source)
    .transform_window(window_sum_amount=amount.sum(), window_lead_label=label.shift(1))
    .transform_calculate(
        calc_lead=(
            nw.when(window_lead_label.is_null()).then(label).otherwise(window_lead_label)
        ),
        calc_prev_sum=calc_prev_sum,
        calc_amount=calc_amount,
        calc_text_amount=(
            nw.when(~begin_or_end, calc_amount > 0).then(nw.lit("+")) + calc_amount
        ),
        calc_center=(window_sum_amount + calc_prev_sum) / 2,
        calc_sum_dec=(nw.when(window_sum_amount < calc_prev_sum).then(window_sum_amount)),
        calc_sum_inc=(nw.when(window_sum_amount > calc_prev_sum).then(window_sum_amount)),
    )
    .encode(x=api.X("label:O", axis=api.Axis(title="Months", labelAngle=0), sort=None))
)


color = (
    nw.when(begin_or_end)
    .then(nw.lit("#878d96"))
    .when(calc_amount < 0)
    .then(nw.lit("#fa4d56"))
    .otherwise(nw.lit("#24a148"))
)


bar = base.mark_bar(size=45).encode(
    y=api.Y("calc_prev_sum:Q", title="Amount"), y2="window_sum_amount:Q", color=color
)

# The "rule" chart is for the horizontal lines that connect the bars
rule = base.mark_rule(xOffset=-22.5, x2Offset=22.5).encode(
    y="window_sum_amount:Q", x2="calc_lead"
)


chart = api.layer(
    bar,
    rule,
    base.mark_text(baseline="bottom", dy=-4).encode(
        text="calc_sum_inc:N", y="calc_sum_inc:Q"
    ),
    base.mark_text(baseline="top", dy=4).encode(
        text="calc_sum_dec:N", y="calc_sum_dec:Q"
    ),
    base.mark_text(baseline="middle").encode(
        text="calc_text_amount:N", y="calc_center:Q", color=nw.lit("white")
    ),
    width=800,
    height=450,
).to_altair()
