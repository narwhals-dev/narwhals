# (https://altair-viz.github.io/gallery/waterfall_chart.html)
from __future__ import annotations

import polars as pl

import narwhals._plan as nw
from narwhals._plan.altair.aggregate import window_transform
from narwhals._plan.altair.calculate import calculate_transform

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
begin = nw.lit("Begin")
end = nw.lit("End")
when_end = nw.when(label == end)
empty = nw.lit("")


# NOTE: Need to be outside the altair api for now - but it is legit!
tf_window = window_transform(
    window_sum_amount=amount.sum(), window_lead_label=label.shift(1)
)


calc_prev_sum = when_end.then(0).otherwise(window_sum_amount - amount)
calc_amount = when_end.then(window_sum_amount).otherwise(amount)


tf_calculate = calculate_transform(
    calc_lead=(
        nw.when(window_lead_label.is_null()).then(label).otherwise(window_lead_label)
    ),
    calc_prev_sum=calc_prev_sum,
    calc_amount=calc_amount,
    calc_text_amount=(
        nw.when(~label.is_in(["Begin", "End"]), calc_amount > 0).then(nw.lit("+"))
        + calc_amount
    ),
    calc_center=(window_sum_amount + calc_prev_sum) / 2,
    calc_sum_dec=(
        nw.when(window_sum_amount < calc_prev_sum)
        .then(window_sum_amount)
        .otherwise(empty)
    ),
    calc_sum_inc=(
        nw.when(window_sum_amount > calc_prev_sum)
        .then(window_sum_amount)
        .otherwise(empty)
    ),
)
