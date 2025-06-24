from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals._plan.demo as nwd
from tests.utils import POLARS_VERSION

if TYPE_CHECKING:
    from narwhals._plan.dummy import DummyExpr

pytest.importorskip("polars")
import polars as pl

if POLARS_VERSION >= (1, 0):
    # https://github.com/pola-rs/polars/pull/16743
    OVER_CASE = (
        nwd.col("a").last().over("b", order_by="c"),
        pl.col("a").last().over("b", order_by="c"),
        ["a", "b"],
    )
else:  # pragma: no cover
    OVER_CASE = (nwd.col("a").last().over("b"), pl.col("a").last().over("b"), ["a", "b"])


@pytest.mark.parametrize(
    ("nw_expr", "pl_expr", "expected"),
    [
        (
            nwd.col("a").alias("b").min().alias("c").alias("d"),
            pl.col("a").alias("b").min().alias("c").alias("d"),
            ["a"],
        ),
        (
            (nwd.col("a") + (nwd.col("a") - nwd.col("b"))).alias("c"),
            (pl.col("a") + (pl.col("a") - pl.col("b"))).alias("c"),
            ["a", "a", "b"],
        ),
        OVER_CASE,
        (
            (nwd.col("a", "b", "c").sort().abs() * 20).max(),
            (pl.col("a", "b", "c").sort().abs() * 20).max(),
            [],
        ),
        (nwd.all().mean(), pl.all().mean(), []),
        (nwd.all().mean().sort_by("d"), pl.all().mean().sort_by("d"), ["d"]),
    ],
)
def test_meta_root_names(
    nw_expr: DummyExpr, pl_expr: pl.Expr, expected: list[str]
) -> None:
    pl_result = pl_expr.meta.root_names()
    nw_result = nw_expr.meta.root_names()
    assert nw_result == expected
    assert nw_result == pl_result


@pytest.mark.parametrize(
    ("nw_expr", "pl_expr", "expected"),
    [
        (nwd.col("a"), pl.col("a"), "a"),
        (nwd.lit(1), pl.lit(1), "literal"),
        (nwd.len(), pl.len(), "len"),
        (
            nwd.col("a")
            .alias("b")
            .min()
            .alias("c")
            .over("e", "f")
            .sort_by(nwd.nth(9), nwd.col("g", "h")),
            pl.col("a")
            .alias("b")
            .min()
            .alias("c")
            .over("e", "f")
            .sort_by(pl.nth(9), pl.col("g", "h")),
            "c",
        ),
        pytest.param(
            nwd.col("c").alias("x").fill_null(50),
            pl.col("c").alias("x").fill_null(50),
            "x",
            id="FunctionExpr-Literal",
        ),
    ],
)
def test_meta_output_name(nw_expr: DummyExpr, pl_expr: pl.Expr, expected: str) -> None:
    pl_result = pl_expr.meta.output_name()
    nw_result = nw_expr.meta.output_name()
    assert nw_result == expected
    assert nw_result == pl_result
