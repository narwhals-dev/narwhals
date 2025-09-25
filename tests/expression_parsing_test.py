from __future__ import annotations

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError
from tests.utils import POLARS_VERSION, Constructor, assert_equal_data

pytest.importorskip("polars")

import polars as pl


@pytest.mark.parametrize(
    ("expr", "pl_expr", "expected"),
    [
        (nw.col("a"), pl.col("a"), [-1, 2, 3]),
        (nw.col("a").mean(), pl.col("a").mean(), [4 / 3, 4 / 3, 4 / 3]),
        (
            nw.col("a").cum_sum().over(order_by="i"),
            pl.col("a").cum_sum().over(order_by="i"),
            [-1, 1, 4],
        ),
        (
            nw.col("a").cum_sum().abs().over(order_by="i"),
            pl.col("a").cum_sum().abs().over(order_by="i"),
            [1, 1, 4],
        ),
        (
            (nw.col("a").cum_sum() + 1).over(order_by="i"),
            (pl.col("a").cum_sum() + 1).over(order_by="i"),
            [0, 2, 5],
        ),
        (
            nw.sum_horizontal(nw.col("a"), nw.col("a").cum_sum()).over(order_by="a"),
            pl.sum_horizontal(pl.col("a"), pl.col("a").cum_sum()).over(order_by="a"),
            [-2, 3, 7],
        ),
        (
            nw.sum_horizontal(nw.col("a"), nw.col("a").cum_sum().over(order_by="i")),
            pl.sum_horizontal(pl.col("a"), pl.col("a").cum_sum().over(order_by="i")),
            [-2, 3, 7],
        ),
        (
            nw.sum_horizontal(nw.col("a").diff(), nw.col("a").cum_sum()).over(
                order_by="i"
            ),
            pl.sum_horizontal(pl.col("a").diff(), pl.col("a").cum_sum()).over(
                order_by="i"
            ),
            [-1.0, 4.0, 5.0],
        ),
        (
            nw.sum_horizontal(nw.col("a").diff().abs(), nw.col("a").cum_sum()).over(
                order_by="i"
            ),
            pl.sum_horizontal(pl.col("a").diff().abs(), pl.col("a").cum_sum()).over(
                order_by="i"
            ),
            [-1.0, 4.0, 5.0],
        ),
        (
            (nw.col("a").sum() + nw.col("a").rolling_sum(2, min_samples=1)).over(
                order_by="i"
            ),
            (pl.col("a").sum() + pl.col("a").rolling_sum(2, min_samples=1)).over(
                order_by="i"
            ),
            [3.0, 5.0, 9.0],
        ),
        (
            (nw.col("a").sum() + nw.col("a").mean()).over("b"),
            (pl.col("a").sum() + pl.col("a").mean()).over("b"),
            [1.5, 1.5, 6.0],
        ),
        (
            (nw.col("a").mean().abs() + nw.sum_horizontal(nw.col("a").diff())).over(
                order_by="i"
            ),
            (pl.col("a").mean().abs() + pl.sum_horizontal(pl.col("a").diff())).over(
                order_by="i"
            ),
            [4 / 3, 13 / 3, 7 / 3],
        ),
    ],
)
def test_over_pushdown(
    constructor: Constructor, expr: nw.Expr, pl_expr: pl.Expr, expected: list[float]
) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 10):
        pytest.skip()
    data = {"a": [-1, 2, 3], "b": [1, 1, 2], "i": [0, 1, 2]}
    df = nw.from_native(constructor(data)).lazy()
    result = df.select("i", a=expr).sort("i").select("a")
    assert_equal_data(result, {"a": expected})

    # Confirm that doing the calculation in pure-Polars produces the same result.
    pl_result = {"a": pl.DataFrame(data).select("i", a=pl_expr).sort("i")["a"].to_list()}
    assert_equal_data(result, pl_result)


@pytest.mark.parametrize(
    "expr",
    [
        nw.col("a").cum_sum(),
        nw.col("a").cum_sum().cum_sum().over(order_by="i"),
        nw.col("a").cum_sum().cum_sum(),
        nw.sum_horizontal(nw.col("a"), nw.col("a").cum_sum()),
        nw.sum_horizontal(nw.col("a").diff(), nw.col("a").cum_sum().over(order_by="i")),
        nw.col("a").mean().over(order_by="i"),
        nw.col("a").mean().over("b").over("c"),
        nw.col("a").mean().over("b").over("c", order_by="i"),
        nw.col("a").mean().mean(),
        nw.col("a").mean().sum(),
        nw.col("a").mean().drop_nulls(),
        nw.col("a").mean().rank(),
        nw.col("a").mean().is_unique(),
        nw.col("a").mean().diff(),
        nw.col("a").fill_null(3).over("b"),
        nw.col("a").drop_nulls().over("b"),
        nw.col("a").drop_nulls().over("b", order_by="i"),
        nw.col("a").diff().drop_nulls().over("b", order_by="i"),
    ],
)
def test_invalid_operations(constructor: Constructor, expr: nw.Expr) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 10):
        pytest.skip()
    df = nw.from_native(
        constructor({"a": [-1, 2, 3], "b": [1, 1, 1], "c": [2, 2, 2], "i": [0, 1, 2]})
    ).lazy()
    with pytest.raises((InvalidOperationError, NotImplementedError)):
        df.select(a=expr)
