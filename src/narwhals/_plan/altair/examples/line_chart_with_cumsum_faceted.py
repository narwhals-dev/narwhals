"""[Faceted Line Chart with Cumulative Sum].

[Faceted Line Chart with Cumulative Sum]: https://altair-viz.github.io/gallery/line_chart_with_cumsum_faceted.html
"""

from __future__ import annotations

import narwhals._plan as nw
from narwhals._plan.altair import api

source = api.datasets.load("disasters", backend="polars")
columns_sorted = ["Drought", "Epidemic", "Earthquake", "Flood"]

chart = (
    api.Chart(source)
    .transform_filter(
        nw.col("Entity").is_in(columns_sorted), nw.col("Year").is_between(1900, 2000)
    )
    .transform_window(cumulative_deaths=nw.col("Deaths").cum_sum().over("Entity"))
    .mark_line()
    .encode(
        api.X("Year", title=None).axis(format="d"),
        api.Y("cumulative_deaths:Q", title=None),
        api.Color("Entity", legend=None),
    )
    .properties(width=300, height=150)
    # TODO @dangotbanned: Consider re-wrapping `Title` as it can take `Parameter`s
    .facet(
        api.Facet(
            "Entity",
            title=None,
            sort=columns_sorted,
            header=api.Header(labelAnchor="start", labelFontStyle="italic"),
        ),
        title=api.Title(
            text=["Cumulative casualties by type of disaster", "in the 20th century"],
            anchor="middle",
        ),
        columns=2,
    )
    .resolve_axis(y="independent", x="independent")
)
