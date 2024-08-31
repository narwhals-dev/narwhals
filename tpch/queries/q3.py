from datetime import date
from datetime import datetime
from typing import Any

import ibis

import narwhals as nw


def query_pandas_native(
    customer_ds: Any,
    line_item_ds: Any,
    orders_ds: Any,
) -> Any:
    var1 = "BUILDING"
    var2 = date(1995, 3, 15)

    fcustomer = customer_ds[customer_ds["c_mktsegment"] == var1]

    jn1 = fcustomer.merge(orders_ds, left_on="c_custkey", right_on="o_custkey")
    jn2 = jn1.merge(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")

    jn2 = jn2[jn2["o_orderdate"] < var2]
    jn2 = jn2[jn2["l_shipdate"] > var2]
    jn2["revenue"] = jn2.l_extendedprice * (1 - jn2.l_discount)

    gb = jn2.groupby(["o_orderkey", "o_orderdate", "o_shippriority"], as_index=False)
    agg = gb["revenue"].sum()

    sel = agg.loc[:, ["o_orderkey", "revenue", "o_orderdate", "o_shippriority"]]
    sel = sel.rename({"o_orderkey": "l_orderkey"}, axis="columns")

    sorted = sel.sort_values(by=["revenue", "o_orderdate"], ascending=[False, True])

    return sorted.head(10)  # type: ignore[no-any-return]


def query(
    customer_ds_raw: Any,
    line_item_ds_raw: Any,
    orders_ds_raw: Any,
) -> Any:
    var_1 = var_2 = datetime(1995, 3, 15)
    var_3 = "BUILDING"

    customer_ds = nw.from_native(customer_ds_raw)
    line_item_ds = nw.from_native(line_item_ds_raw)
    orders_ds = nw.from_native(orders_ds_raw)

    q_final = (
        customer_ds.filter(nw.col("c_mktsegment") == var_3)
        .join(orders_ds, left_on="c_custkey", right_on="o_custkey")
        .join(line_item_ds, left_on="o_orderkey", right_on="l_orderkey")
        .filter(
            nw.col("o_orderdate") < var_2,
            nw.col("l_shipdate") > var_1,
        )
        .with_columns(
            (nw.col("l_extendedprice") * (1 - nw.col("l_discount"))).alias("revenue")
        )
        .group_by(["o_orderkey", "o_orderdate", "o_shippriority"])
        .agg([nw.sum("revenue")])
        .select(
            [
                nw.col("o_orderkey").alias("l_orderkey"),
                "revenue",
                "o_orderdate",
                "o_shippriority",
            ]
        )
        .sort(by=["revenue", "o_orderdate"], descending=[True, False])
        .head(10)
    )

    return nw.to_native(q_final)


def query_ibis(
    customer: Any,
    lineitem: Any,
    orders: Any,
    *,
    tool: str,
) -> Any:
    var1 = "BUILDING"
    var2 = date(1995, 3, 15)

    q_final = (
        customer.filter(customer["c_mktsegment"] == var1)
        .join(orders, customer["c_custkey"] == orders["o_custkey"])
        .join(lineitem, orders["o_orderkey"] == lineitem["l_orderkey"])
        .filter(ibis._["o_orderdate"] < var2)
        .filter(ibis._["l_shipdate"] > var2)
        .mutate(revenue=(lineitem["l_extendedprice"] * (1 - lineitem["l_discount"])))
        .group_by(
            "o_orderkey",
            "o_orderdate",
            "o_shippriority",
        )
        .agg(revenue=ibis._["revenue"].sum())
        .select(
            ibis._["o_orderkey"].name("o_orderkey"),
            "revenue",
            "o_orderdate",
            "o_shippriority",
        )
        .order_by(ibis.desc("revenue"), "o_orderdate")
        .limit(10)
    )
    if tool == "pandas":
        return q_final.to_pandas()
    if tool == "polars":
        return q_final.to_polars()
    msg = "expected pandas or polars"
    raise ValueError(msg)
