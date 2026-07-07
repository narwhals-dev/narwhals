"""[Interactive Chart with Aggregation].

[Interactive Chart with Aggregation]: https://altair-viz.github.io/gallery/interactive_aggregation.html
"""

from __future__ import annotations

import altair as alt
from altair.datasets import load

import narwhals._plan as nw

# TODO @dangotbanned: Move the impl modules under a subpackage (next to `examples`)
# - export `Chart`, `param`, `selection_interval`, `selection_point`, etc
# - import that here as `api`
from narwhals._plan.altair import chart as nw_alt
from narwhals._plan.altair.parameter import param

base = nw_alt.Chart(load("movies", backend="polars")).mark_circle()
threshold = param("threshold", value=5, bind=alt.binding_range(min=0, max=10, step=0.1))

imdb = nw.col("IMDB Rating")
tomatoes = nw.col("Rotten Tomatoes Rating")

chart = nw_alt.layer(
    base.encode(x=imdb.name.keep(), y=tomatoes.name.keep()).transform_filter(
        imdb >= threshold
    ),
    base.encode(
        x=imdb.hist(bin_count=10),
        y=tomatoes.hist(bin_count=10),
        size=nw.len().clip(0, 160),
    ).transform_filter(imdb < threshold),
    nw_alt.Chart().mark_rule(color="gray").encode(x=threshold, strokeWidth=nw.lit(6)),
).add_params(threshold)
