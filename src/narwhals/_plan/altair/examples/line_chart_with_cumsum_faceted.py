"""[Faceted Line Chart with Cumulative Sum].

[Faceted Line Chart with Cumulative Sum]: https://altair-viz.github.io/gallery/line_chart_with_cumsum_faceted.html
"""

from __future__ import annotations

import altair as alt

from narwhals._plan.altair import api

source = api.datasets.load("disasters", backend="polars")
columns_sorted = ["Drought", "Epidemic", "Earthquake", "Flood"]

chart = (
    alt.Chart(source)
    .transform_filter(
        # TODO @dangotbanned: Use `col("Entity").is_in(columns_sorted)`
        alt.FieldOneOfPredicate(field="Entity", oneOf=columns_sorted),
        # TODO @dangotbanned: Use `col("Year").is_between(1900, 2000)`
        alt.FieldRangePredicate(field="Year", range=[1900, 2000]),
    )
    # TODO @dangotbanned: Multiple gaps, goal is `col("Deaths").cum_sum().over("Entity")`
    # - [ ] Support `over` here (like `transform_aggregate`)
    # - [ ] Support `cum_*` functions in `transform_window` only
    #   - They need to split out with `frame=(None, 0)`
    .transform_window(cumulative_deaths="sum(Deaths)", groupby=["Entity"])
    .mark_line()
    .encode(
        alt.X("Year:Q", title=None).axis(format="d"),
        alt.Y("cumulative_deaths:Q", title=None),
        alt.Color("Entity:N", legend=None),
    )
    .properties(width=300, height=150)
    # TODO @dangotbanned: Add `Chart.facet`
    # TODO @dangotbanned: Re-export `Facet`, `Header`, `Title` (although )
    # TODO @dangotbanned: Consider re-wrapping `Title` as it can take `Parameter`s
    .facet(
        facet=alt.Facet(
            "Entity:N",
            title=None,
            sort=columns_sorted,
            header=alt.Header(labelAnchor="start", labelFontStyle="italic"),
        ),
        title=alt.Title(
            text=["Cumulative casualties by type of disaster", "in the 20th century"],
            anchor="middle",
        ),
        columns=2,
    )
    .resolve_axis(y="independent", x="independent")
)
