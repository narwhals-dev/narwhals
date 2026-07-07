"""[Wheat and Wages].

[Wheat and Wages]: https://altair-viz.github.io/gallery/wheat_wages.html
"""

from __future__ import annotations

import polars as pl

import narwhals._plan as nw
from narwhals._plan.altair import api

load = api.datasets.Loader.from_backend("polars")

year, index, start, end = (nw.col(name) for name in ("year", "index", "start", "end"))
base_wheat = api.Chart(load("wheat")).transform_calculate(year_end=year + 5)
bars = base_wheat.mark_bar(fill="#aaa", stroke="#999").encode(
    api.X("year").bin("binned").axis(format="d", tickCount=5).scale(zero=False),
    api.Y("wheat").axis(zindex=1),
    x2="year_end",
)

section_line = (
    api.Chart(pl.DataFrame({"year": [1600, 1650, 1700, 1750, 1800]}))
    .mark_rule(stroke="#000", strokeWidth=0.6, opacity=0.7)
    .encode(x=year)
)

area = base_wheat.mark_area(color="#a4cedb", opacity=0.7).encode(x=year, y="wages")
area_line_1 = area.mark_line(color="#000", opacity=0.7)
area_line_2 = area.mark_line(yOffset=-2, color="#EE8182")

cond = nw.when(nw.col("commonwealth").is_null(), index % 2).then(-1).otherwise(1)
base_monarchs = api.Chart(load("monarchs")).transform_calculate(
    x=start + (end - start) / 2, y=nw.lit(95), offset=cond * 2 + 95, off2=cond + 95
)
top_bars = base_monarchs.mark_bar(stroke="#000").encode(
    api.Fill("commonwealth").legend(None).scale(range=["black", "white"]),
    x=start,
    x2=end,
    y="y:Q",
    y2="offset",
)

top_text = base_monarchs.mark_text(yOffset=14, fontSize=9, fontStyle="italic").encode(
    x="x:Q", y="off2:Q", text="name"
)

chart = (
    api.layer(bars, section_line, area, area_line_1, area_line_2, top_bars, top_text)
    .properties(width=900, height=400)
    .configure_axis(title=None, gridColor="white", gridOpacity=0.25, domain=False)
    .configure_view(stroke="transparent")
)
