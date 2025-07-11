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
if POLARS_VERSION >= (0, 20, 5):
    LEN_CASE = (nwd.len(), pl.len(), "len")
else:  # pragma: no cover
    LEN_CASE = (nwd.len().alias("count"), pl.count(), "count")


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
        LEN_CASE,
        pytest.param(
            (
                nwd.col("a")
                .alias("b")
                .min()
                .alias("c")
                .over("e", "f")
                .sort_by(nwd.col("i"), nwd.col("g", "h"))
            ),
            (
                pl.col("a")
                .alias("b")
                .min()
                .alias("c")
                .over("e", "f")
                .sort_by(pl.col("i"), pl.col("g", "h"))
            ),
            "c",
            id="Kitchen-Sink",
        ),
        pytest.param(
            nwd.col("c").alias("x").fill_null(50),
            pl.col("c").alias("x").fill_null(50),
            "x",
            id="FunctionExpr-Literal",
        ),
        pytest.param(
            (
                nwd.col("ROOT")
                .alias("ROOT-ALIAS")
                .filter(nwd.col("b") >= 30, nwd.col("c").alias("d") == 7)
                + nwd.col("RHS").alias("RHS-ALIAS")
            ),
            (
                pl.col("ROOT")
                .alias("ROOT-ALIAS")
                .filter(pl.col("b") >= 30, pl.col("c").alias("d") == 7)
                + pl.col("RHS").alias("RHS-ALIAS")
            ),
            "ROOT-ALIAS",
            id="BinaryExpr-Multiple",
        ),
        pytest.param(
            nwd.col("ROOT").alias("ROOT-ALIAS").mean().over(nwd.col("a").alias("b")),
            pl.col("ROOT").alias("ROOT-ALIAS").mean().over(pl.col("a").alias("b")),
            "ROOT-ALIAS",
            id="WindowExpr",
        ),
        pytest.param(
            nwd.when(nwd.col("a").alias("a?")).then(10),
            pl.when(pl.col("a").alias("a?")).then(10),
            "literal",
            id="When-Literal",
        ),
        pytest.param(
            nwd.when(nwd.col("a").alias("a?")).then(nwd.col("b")).otherwise(20),
            pl.when(pl.col("a").alias("a?")).then(pl.col("b")).otherwise(20),
            "b",
            id="When-Column-Literal",
        ),
        pytest.param(
            nwd.when(a=1).then(10).otherwise(nwd.col("c").alias("c?")),
            pl.when(a=1).then(10).otherwise(pl.col("c").alias("c?")),
            "literal",
            id="When-Literal-Alias",
        ),
        pytest.param(
            (
                nwd.when(nwd.col("a").alias("a?"))
                .then(1)
                .when(nwd.col("b") == 1)
                .then(nwd.col("c"))
            ),
            (
                pl.when(pl.col("a").alias("a?"))
                .then(1)
                .when(pl.col("b") == 1)
                .then(pl.col("c"))
            ),
            "literal",
            id="When-Literal-BinaryExpr-Column",
        ),
        pytest.param(
            (
                nwd.when(nwd.col("foo") > 2, nwd.col("bar") < 3)
                .then(nwd.lit("Yes"))
                .otherwise(nwd.lit("No"))
                .alias("TARGET")
            ),
            (
                pl.when(pl.col("foo") > 2, pl.col("bar") < 3)
                .then(pl.lit("Yes"))
                .otherwise(pl.lit("No"))
                .alias("TARGET")
            ),
            "TARGET",
            id="When2-Literal-Literal-Alias",
        ),
        pytest.param(
            (nwd.col("ROOT").alias("ROOT-ALIAS").filter(nwd.col("c") <= 1).mean()),
            (pl.col("ROOT").alias("ROOT-ALIAS").filter(pl.col("c") <= 1).mean()),
            "ROOT-ALIAS",
            id="Filter",
        ),
        pytest.param(
            nwd.int_range(0, 10), pl.int_range(0, 10), "literal", id="IntRange-Literal"
        ),
        pytest.param(
            nwd.int_range(nwd.col("b").first(), 10),
            pl.int_range(pl.col("b").first(), 10),
            "b",
            id="IntRange-Column",
        ),
    ],
)
def test_meta_output_name(nw_expr: DummyExpr, pl_expr: pl.Expr, expected: str) -> None:
    pl_result = pl_expr.meta.output_name()
    nw_result = nw_expr.meta.output_name()
    assert nw_result == expected
    assert nw_result == pl_result
