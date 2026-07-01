# TODO @dangotbanned: narwhalify!
"""[Wheat and Wages].

[Wheat and Wages]: https://altair-viz.github.io/gallery/wheat_wages.html
"""

from __future__ import annotations

import altair as alt
import polars as pl
from altair.datasets import Loader

load = Loader.from_backend("polars")

base_wheat = alt.Chart(load("wheat")).transform_calculate(year_end="+datum.year + 5")

base_monarchs = alt.Chart(load("monarchs")).transform_calculate(
    offset="((!datum.commonwealth && datum.index % 2) ? -1: 1) * 2 + 95",
    off2="((!datum.commonwealth && datum.index % 2) ? -1: 1) + 95",
    y="95",
    x="+datum.start + (+datum.end - +datum.start)/2",
)

bars = base_wheat.mark_bar(fill="#aaa", stroke="#999").encode(
    alt.X("year").bin("binned").axis(format="d", tickCount=5).scale(zero=False),
    alt.Y("wheat").axis(zindex=1),
    x2="year_end",
)

section_line = (
    alt.Chart(pl.DataFrame({"year": [1600, 1650, 1700, 1750, 1800]}))
    .mark_rule(stroke="#000", strokeWidth=0.6, opacity=0.7)
    .encode(x="year")
)

area = base_wheat.mark_area(color="#a4cedb", opacity=0.7).encode(x="year", y="wages")
area_line_1 = area.mark_line(color="#000", opacity=0.7)
area_line_2 = area.mark_line(yOffset=-2, color="#EE8182")

top_bars = base_monarchs.mark_bar(stroke="#000").encode(
    alt.Fill("commonwealth").legend(None).scale(range=["black", "white"]),
    x="start",
    x2="end",
    y="y:Q",
    y2="offset",
)

top_text = base_monarchs.mark_text(yOffset=14, fontSize=9, fontStyle="italic").encode(
    x="x:Q", y="off2:Q", text="name"
)

chart = (
    alt.layer(bars, section_line, area, area_line_1, area_line_2, top_bars, top_text)
    .properties(width=900, height=400)
    .configure_axis(title=None, gridColor="white", gridOpacity=0.25, domain=False)
    .configure_view(stroke="transparent")
)
