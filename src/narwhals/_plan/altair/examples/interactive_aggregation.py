"""[Interactive Chart with Aggregation].

[Interactive Chart with Aggregation]: https://altair-viz.github.io/gallery/interactive_aggregation.html
"""

from __future__ import annotations

import altair as alt
from altair.datasets import Loader

import narwhals._plan as nw

# TODO @dangotbanned: Move the impl modules under a subpackage (next to `examples`)
# - export `Chart`, `param`, `selection_interval`, `selection_point`, etc
# - import that here as `api`
from narwhals._plan.altair import chart as nw_alt
from narwhals._plan.altair.parameter import param

load = Loader.from_backend("polars")
base = nw_alt.Chart(load("movies")).mark_circle()
threshold = param("threshold", value=5, bind=alt.binding_range(min=0, max=10, step=0.1))

imdb_rating = nw.col("IMDB Rating")
rt_rating = nw.col("Rotten Tomatoes Rating")

raw = base.encode(x=imdb_rating.name.keep(), y=rt_rating.name.keep()).transform_filter(
    imdb_rating >= threshold
)

aggregated = base.encode(
    x=imdb_rating.hist(bin_count=10),
    y=rt_rating.hist(bin_count=10),
    size=nw.len().clip(0, 160),
).transform_filter(imdb_rating < threshold)

rule = nw_alt.Chart().mark_rule(color="gray").encode(x=threshold, strokeWidth=nw.lit(6))

chart = nw_alt.layer(raw, aggregated, rule).add_params(threshold)
