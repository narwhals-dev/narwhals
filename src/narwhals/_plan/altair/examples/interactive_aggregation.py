"""[Interactive Chart with Aggregation].

[Interactive Chart with Aggregation]: https://altair-viz.github.io/gallery/interactive_aggregation.html
"""

from __future__ import annotations

import altair as alt
from altair.datasets import Loader

load = Loader.from_backend("polars")
base = alt.Chart(load("movies")).mark_circle()
threshold = alt.param("threshold", 5, bind=alt.binding_range(min=0, max=10, step=0.1))

raw = base.encode(
    x=alt.X("IMDB Rating").title("IMDB Rating"),
    y=alt.Y("Rotten Tomatoes Rating").title("Rotten Tomatoes Rating"),
).transform_filter(alt.datum["IMDB Rating"] >= threshold)

aggregated = base.encode(
    x=alt.X("IMDB Rating").bin(maxbins=10),
    y=alt.Y("Rotten Tomatoes Rating").bin(maxbins=10),
    size=alt.Size("count()").scale(domain=[0, 160]),
).transform_filter(alt.datum["IMDB Rating"] < threshold)

rule = (
    alt.Chart()
    .mark_rule(color="gray")
    .encode(
        strokeWidth=alt.value(6),
        x=alt.X(datum=alt.expr(threshold.name), type="quantitative"),
    )
)

chart = alt.layer(raw, aggregated, rule).add_params(threshold)
