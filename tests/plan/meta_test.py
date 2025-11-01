from __future__ import annotations

import pytest

from narwhals import _plan as nwp
from tests.utils import POLARS_VERSION

pytest.importorskip("polars")
import polars as pl

if POLARS_VERSION >= (1, 0):
    # https://github.com/pola-rs/polars/pull/16743
    OVER_CASE = (
        nwp.col("a").last().over("b", order_by="c"),
        pl.col("a").last().over("b", order_by="c"),
        ["a", "b"],
    )
else:  # pragma: no cover
    OVER_CASE = (nwp.col("a").last().over("b"), pl.col("a").last().over("b"), ["a", "b"])
if POLARS_VERSION >= (0, 20, 5):
    LEN_CASE = (nwp.len(), pl.len(), "len")
else:  # pragma: no cover
    LEN_CASE = (nwp.len().alias("count"), pl.count(), "count")


@pytest.mark.parametrize(
    ("nw_expr", "pl_expr", "expected"),
    [
        (
            nwp.col("a").alias("b").min().alias("c").alias("d"),
            pl.col("a").alias("b").min().alias("c").alias("d"),
            ["a"],
        ),
        (
            (nwp.col("a") + (nwp.col("a") - nwp.col("b"))).alias("c"),
            (pl.col("a") + (pl.col("a") - pl.col("b"))).alias("c"),
            ["a", "a", "b"],
        ),
        OVER_CASE,
        (
            (nwp.col("a", "b", "c").sort().abs() * 20).max(),
            (pl.col("a", "b", "c").sort().abs() * 20).max(),
            [],
        ),
        (nwp.all().mean(), pl.all().mean(), []),
        (nwp.all().mean().sort_by("d"), pl.all().mean().sort_by("d"), ["d"]),
    ],
)
def test_meta_root_names(
    nw_expr: nwp.Expr, pl_expr: pl.Expr, expected: list[str]
) -> None:
    pl_result = pl_expr.meta.root_names()
    nw_result = nw_expr.meta.root_names()
    assert nw_result == expected
    assert nw_result == pl_result


@pytest.mark.parametrize(
    ("nw_expr", "pl_expr", "expected"),
    [
        (nwp.col("a"), pl.col("a"), "a"),
        (nwp.lit(1), pl.lit(1), "literal"),
        LEN_CASE,
        pytest.param(
            (
                nwp.col("a")
                .alias("b")
                .min()
                .alias("c")
                .over("e", "f")
                .sort_by(nwp.col("i"), nwp.col("g", "h"))
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
            nwp.col("c").alias("x").fill_null(50),
            pl.col("c").alias("x").fill_null(50),
            "x",
            id="FunctionExpr-Literal",
        ),
        pytest.param(
            (
                nwp.col("ROOT")
                .alias("ROOT-ALIAS")
                .filter(nwp.col("b") >= 30, nwp.col("c").alias("d") == 7)
                + nwp.col("RHS").alias("RHS-ALIAS")
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
            nwp.col("ROOT").alias("ROOT-ALIAS").mean().over(nwp.col("a").alias("b")),
            pl.col("ROOT").alias("ROOT-ALIAS").mean().over(pl.col("a").alias("b")),
            "ROOT-ALIAS",
            id="WindowExpr",
        ),
        pytest.param(
            nwp.when(nwp.col("a").alias("a?")).then(10),
            pl.when(pl.col("a").alias("a?")).then(10),
            "literal",
            id="When-Literal",
        ),
        pytest.param(
            nwp.when(nwp.col("a").alias("a?")).then(nwp.col("b")).otherwise(20),
            pl.when(pl.col("a").alias("a?")).then(pl.col("b")).otherwise(20),
            "b",
            id="When-Column-Literal",
        ),
        pytest.param(
            nwp.when(a=1).then(10).otherwise(nwp.col("c").alias("c?")),
            pl.when(a=1).then(10).otherwise(pl.col("c").alias("c?")),
            "literal",
            id="When-Literal-Alias",
        ),
        pytest.param(
            (
                nwp.when(nwp.col("a").alias("a?"))
                .then(1)
                .when(nwp.col("b") == 1)
                .then(nwp.col("c"))
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
                nwp.when(nwp.col("foo") > 2, nwp.col("bar") < 3)
                .then(nwp.lit("Yes"))
                .otherwise(nwp.lit("No"))
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
            (nwp.col("ROOT").alias("ROOT-ALIAS").filter(nwp.col("c") <= 1).mean()),
            (pl.col("ROOT").alias("ROOT-ALIAS").filter(pl.col("c") <= 1).mean()),
            "ROOT-ALIAS",
            id="Filter",
        ),
        pytest.param(
            nwp.int_range(0, 10), pl.int_range(0, 10), "literal", id="IntRange-Literal"
        ),
        pytest.param(
            nwp.int_range(nwp.col("b").first(), 10),
            pl.int_range(pl.col("b").first(), 10),
            "b",
            id="IntRange-Column",
        ),
    ],
)
def test_meta_output_name(nw_expr: nwp.Expr, pl_expr: pl.Expr, expected: str) -> None:
    pl_result = pl_expr.meta.output_name()
    nw_result = nw_expr.meta.output_name()
    assert nw_result == expected
    assert nw_result == pl_result
