"""[Interactive Chart with Aggregation].

[Interactive Chart with Aggregation]: https://altair-viz.github.io/gallery/interactive_aggregation.html
"""

from __future__ import annotations

import narwhals._plan as nw
from narwhals._plan.altair import api

source = api.datasets.load("movies", backend="polars")
base = api.Chart(source).mark_circle()
threshold = api.param(
    "threshold", value=5, bind=api.binding_range(min=0, max=10, step=0.1)
)

imdb = nw.col("IMDB Rating")
tomatoes = nw.col("Rotten Tomatoes Rating")

chart = api.layer(
    base.encode(x=imdb.name.keep(), y=tomatoes.name.keep()).transform_filter(
        imdb >= threshold
    ),
    base.encode(
        x=imdb.hist(bin_count=10),
        y=tomatoes.hist(bin_count=10),
        size=nw.len().clip(0, 160),
    ).transform_filter(imdb < threshold),
    api.Chart().mark_rule(color="gray").encode(x=threshold, strokeWidth=nw.lit(6)),
).add_params(threshold)
