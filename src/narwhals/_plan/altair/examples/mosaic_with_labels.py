"""[Mosaic Chart with Labels].

[Mosaic Chart with Labels]: https://altair-viz.github.io/gallery/mosaic_with_labels.html
"""

from __future__ import annotations

import altair as alt
from altair.datasets import load

import narwhals._plan as nw
from narwhals._plan.altair import chart as nw_alt

rank_cylinders = nw.col("rank_Cylinders")
rank_origin = nw.col("rank_Origin")
distinct_cylinders = nw.col("distinct_Cylinders")

base = (
    nw_alt.Chart(load("cars", backend="polars"))
    .transform_aggregate(nw.len().over("Origin", "Cylinders"))
    .transform_stack(
        stack="len",
        as_=["stack_count_Origin1", "stack_count_Origin2"],
        offset="normalize",
        sort=[alt.SortField("Origin", "ascending")],
    )
    # TODO @dangotbanned: Add a similar API to `transform_aggregate`
    # - but this one is allowed to use `OverOrdered`
    .transform_window(
        x=nw.min("stack_count_Origin1"),
        x2=nw.max("stack_count_Origin2"),
        rank_Cylinders=nw.col("Cylinders").rank("dense"),
        distinct_Cylinders=nw.col("Cylinders").n_unique(),
        groupby=["Origin"],
        frame=[None, None],
        sort=[alt.SortField("Cylinders", "ascending")],
    )
    # TODO @dangotbanned: Maybe able to merge into the above call?
    .transform_window(
        rank_Origin=nw.col("Origin").rank("dense"),
        frame=[None, None],
        sort=[alt.SortField("Origin", "ascending")],
    )
    .transform_stack(
        stack="len",
        groupby=["Origin"],
        as_=["y", "y2"],
        offset="normalize",
        sort=[alt.SortField("Cylinders", "ascending")],
    )
    .transform_calculate(
        ny=nw.col("y") + (rank_cylinders - 1) * distinct_cylinders * 0.01 / 3,
        ny2=nw.col("y2") + (rank_cylinders - 1) * distinct_cylinders * 0.01 / 3,
        nx=nw.col("x") + (rank_origin - 1) * 0.01,
        nx2=nw.col("x2") + (rank_origin - 1) * 0.01,
        xc=(nw.col("nx") + nw.col("nx2")) / 2,
        yc=(nw.col("ny") + nw.col("ny2")) / 2,
    )
)


rect = base.mark_rect().encode(
    x=alt.X("nx:Q").axis(None),
    x2="nx2",
    y="ny:Q",
    y2="ny2",
    color=alt.Color("Origin:N").legend(None),
    opacity=alt.Opacity("Cylinders:Q").legend(None),
    tooltip=["Origin:N", "Cylinders:Q"],
)


text = base.mark_text(baseline="middle").encode(
    alt.X("xc:Q").axis(None),
    # NOTE:`y=nwp.col("yc").alias("Cylinders").cast(nw.Int64)`?
    # Probably too long
    alt.Y("yc:Q").title("Cylinders"),
    text="Cylinders:N",
)

origin_labels = base.mark_text(baseline="middle", align="center").encode(
    # TODO @dangotbanned: Almost able to do this with `nw.min("xc").alias("Origin")`
    # The axis orient though may be a blocker
    alt.X("min(xc):Q").title("Origin").axis(orient="top"),
    alt.Color("Origin").legend(None),
    text="Origin",
)

chart = (
    (origin_labels & (rect + text))
    .resolve_scale(x="shared")
    .configure_view(stroke="")
    .configure_concat(spacing=10)
    .configure_axis(domain=False, ticks=False, labels=False, grid=False)
)
