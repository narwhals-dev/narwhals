"""[Interactive Chart with Aggregation].

[Interactive Chart with Aggregation]: https://altair-viz.github.io/gallery/interactive_aggregation.html
"""

from __future__ import annotations

import altair as alt
from altair.datasets import Loader

import narwhals._plan as nw
from narwhals._plan.altair import chart as nw_alt

load = Loader.from_backend("polars")
base = nw_alt.Chart(load("movies")).mark_circle()
threshold = alt.param("threshold", 5, bind=alt.binding_range(min=0, max=10, step=0.1))

raw = base.encode(
    # TODO @dangotbanned: Support these as either:
    # - `nw.col("IMDB Rating").alias("IMDB Rating")`
    # - `nw.col("IMDB Rating").name.keep()`
    #   - This example is an edge case where that makes sense
    x=alt.X("IMDB Rating").title("IMDB Rating"),
    y=alt.Y("Rotten Tomatoes Rating").title("Rotten Tomatoes Rating"),
).transform_filter(alt.datum["IMDB Rating"] >= threshold)

aggregated = base.encode(
    # TODO @dangotbanned: Support as:
    # - `nw.col("IMDB Rating").hist(bin_count=10)`
    # - `nw.col("Rotten Tomatoes Rating").hist(bin_count=10)`
    x=alt.X("IMDB Rating").bin(maxbins=10),
    y=alt.Y("Rotten Tomatoes Rating").bin(maxbins=10),
    # TODO @dangotbanned: Does narwhals have a way of representing this?
    size=alt.Size("count()").scale(domain=[0, 160]),
).transform_filter(alt.datum["IMDB Rating"] < threshold)

rule = (
    nw_alt.Chart()
    .mark_rule(color="gray")
    .encode(
        strokeWidth=nw.lit(6),
        # TODO @dangotbanned: There must be a better way
        x=alt.X(datum=alt.expr(threshold.name), type="quantitative"),
    )
)

# TODO @dangotbanned: Add `add_params`
chart = nw_alt.layer(raw, aggregated, rule).add_params(threshold)  # pyright: ignore[reportAttributeAccessIssue]
