"""Check that running TPC-H queries through Narwhals isn't meaningfully slower than plain pandas.

Adapted from the benchmark notebook referenced in `docs/overhead.md`
(https://www.kaggle.com/code/marcogorelli/narwhals-vs-pandas-overhead-tpc-h-s2), which
compares hand-written pandas queries against their Narwhals equivalents. We don't
require Narwhals to be faster, just that it doesn't add significant overhead.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import pytest

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
import pandas as pd

import narwhals as nw
from tpch import constants
from tpch.queries import q1 as _q1, q2 as _q2, q3 as _q3, q4 as _q4

if TYPE_CHECKING:
    from collections.abc import Callable

    from tpch.typing_ import ScaleFactor

MAX_SLOWDOWN_FACTOR = 1.5
N_REPEATS = 5


def _best_of(fn: Callable[[], object], *, n: int = N_REPEATS) -> float:
    fn()  # warm-up, e.g. to account for any first-call caching
    best = float("inf")
    for _ in range(n):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def _q1_native(lineitem: pd.DataFrame) -> pd.DataFrame:
    var1 = datetime(1998, 9, 2)
    lineitem_filtered = lineitem[lineitem.l_shipdate <= var1]

    # This is lenient towards pandas as normally an optimizer should decide
    # that this could be computed before the groupby aggregation. Other
    # implementations don't enjoy this benefit.
    lineitem_filtered["disc_price"] = lineitem_filtered.l_extendedprice * (
        1 - lineitem_filtered.l_discount
    )
    lineitem_filtered["charge"] = (
        lineitem_filtered.l_extendedprice
        * (1 - lineitem_filtered.l_discount)
        * (1 + lineitem_filtered.l_tax)
    )
    gb = lineitem_filtered.groupby(["l_returnflag", "l_linestatus"], as_index=False)
    total = gb.agg(
        sum_qty=pd.NamedAgg(column="l_quantity", aggfunc="sum"),
        sum_base_price=pd.NamedAgg(column="l_extendedprice", aggfunc="sum"),
        sum_disc_price=pd.NamedAgg(column="disc_price", aggfunc="sum"),
        sum_charge=pd.NamedAgg(column="charge", aggfunc="sum"),
        avg_qty=pd.NamedAgg(column="l_quantity", aggfunc="mean"),
        avg_price=pd.NamedAgg(column="l_extendedprice", aggfunc="mean"),
        avg_disc=pd.NamedAgg(column="l_discount", aggfunc="mean"),
        count_order=pd.NamedAgg(column="l_orderkey", aggfunc="size"),
    )
    return total.sort_values(["l_returnflag", "l_linestatus"])


def _q2_native(
    region_ds: pd.DataFrame,
    nation_ds: pd.DataFrame,
    supplier_ds: pd.DataFrame,
    part_ds: pd.DataFrame,
    part_supp_ds: pd.DataFrame,
) -> pd.DataFrame:
    var1 = 15
    var2 = "BRASS"
    var3 = "EUROPE"

    jn = (
        part_ds.merge(part_supp_ds, left_on="p_partkey", right_on="ps_partkey")
        .merge(supplier_ds, left_on="ps_suppkey", right_on="s_suppkey")
        .merge(nation_ds, left_on="s_nationkey", right_on="n_nationkey")
        .merge(region_ds, left_on="n_regionkey", right_on="r_regionkey")
    )
    jn = jn[jn["p_size"] == var1]
    jn = jn[jn["p_type"].str.endswith(var2)]
    jn = jn[jn["r_name"] == var3]

    gb = jn.groupby("p_partkey", as_index=False)
    agg = gb["ps_supplycost"].min()
    jn2 = agg.merge(jn, on=["p_partkey", "ps_supplycost"])

    sel = jn2.loc[
        :,
        [
            "s_acctbal",
            "s_name",
            "n_name",
            "p_partkey",
            "p_mfgr",
            "s_address",
            "s_phone",
            "s_comment",
        ],
    ]
    sort = sel.sort_values(
        by=["s_acctbal", "n_name", "s_name", "p_partkey"],
        ascending=[False, True, True, True],
    )
    return sort.head(100)


def _q3_native(
    customer_ds: pd.DataFrame, lineitem_ds: pd.DataFrame, orders_ds: pd.DataFrame
) -> pd.DataFrame:
    var1 = "BUILDING"
    var2 = datetime(1995, 3, 15)

    fcustomer = customer_ds[customer_ds["c_mktsegment"] == var1]
    jn1 = fcustomer.merge(orders_ds, left_on="c_custkey", right_on="o_custkey")
    jn2 = jn1.merge(lineitem_ds, left_on="o_orderkey", right_on="l_orderkey")
    jn2 = jn2[jn2["o_orderdate"] < var2]
    jn2 = jn2[jn2["l_shipdate"] > var2]
    jn2["revenue"] = jn2.l_extendedprice * (1 - jn2.l_discount)

    gb = jn2.groupby(["o_orderkey", "o_orderdate", "o_shippriority"], as_index=False)
    agg = cast("pd.DataFrame", gb["revenue"].sum())

    sel = agg.loc[:, ["o_orderkey", "revenue", "o_orderdate", "o_shippriority"]]
    sel = sel.rename({"o_orderkey": "l_orderkey"}, axis="columns")
    sorted_df = sel.sort_values(by=["revenue", "o_orderdate"], ascending=[False, True])
    return sorted_df.head(10)


def _q4_native(lineitem_ds: pd.DataFrame, orders_ds: pd.DataFrame) -> pd.DataFrame:
    var1 = datetime(1993, 7, 1)
    var2 = datetime(1993, 10, 1)

    jn = lineitem_ds.merge(orders_ds, left_on="l_orderkey", right_on="o_orderkey")
    jn = jn[
        (jn["o_orderdate"] < var2)
        & (jn["o_orderdate"] >= var1)
        & (jn["l_commitdate"] < jn["l_receiptdate"])
    ]
    jn = jn.drop_duplicates(subset=["o_orderpriority", "l_orderkey"])

    gb = jn.groupby("o_orderpriority", as_index=False)
    agg = gb.agg(order_count=pd.NamedAgg(column="o_orderkey", aggfunc="count"))
    return agg.sort_values(["o_orderpriority"])


CASES: list[tuple[str, Callable[..., object], Callable[..., object], tuple[str, ...]]] = [
    ("q1", _q1_native, _q1.query, ("lineitem",)),
    ("q2", _q2_native, _q2.query, ("region", "nation", "supplier", "part", "partsupp")),
    ("q3", _q3_native, _q3.query, ("customer", "lineitem", "orders")),
    ("q4", _q4_native, _q4.query, ("lineitem", "orders")),
]


@pytest.fixture(scope="module")
def tables(scale_factor: ScaleFactor) -> dict[str, pd.DataFrame]:
    sf_dir = constants._scale_factor_dir(scale_factor)
    names = {name for _, _, _, table_names in CASES for name in table_names}
    return {
        name: pd.read_parquet(
            sf_dir / f"{name}.parquet", engine="pyarrow", dtype_backend="pyarrow"
        )
        for name in names
    }


@pytest.mark.parametrize(
    ("name", "native_fn", "narwhals_fn", "table_names"), CASES, ids=[c[0] for c in CASES]
)
def test_overhead(
    name: str,
    native_fn: Callable[..., object],
    narwhals_fn: Callable[..., nw.DataFrame[Any]],
    table_names: tuple[str, ...],
    tables: dict[str, pd.DataFrame],
) -> None:
    args = tuple(tables[t] for t in table_names)

    def run_native() -> None:
        native_fn(*args)

    def run_narwhals() -> None:
        narwhals_fn(*(nw.from_native(arg) for arg in args)).to_native()

    native_time = _best_of(run_native)
    narwhals_time = _best_of(run_narwhals)

    threshold = native_time * MAX_SLOWDOWN_FACTOR
    assert narwhals_time <= threshold, (
        f"{name}: pandas via Narwhals took {narwhals_time * 1000:.1f}ms, "
        f"vs {native_time * 1000:.1f}ms for native pandas - more than "
        f"{MAX_SLOWDOWN_FACTOR}x slower (allowed up to {threshold * 1000:.1f}ms). "
        "This suggests a real Narwhals overhead regression, see docs/overhead.md."
    )
