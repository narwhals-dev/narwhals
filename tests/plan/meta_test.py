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
else:
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
