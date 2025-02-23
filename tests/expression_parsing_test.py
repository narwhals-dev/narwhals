from __future__ import annotations

import pytest

import narwhals as nw


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nw.col("a"), 0),
        (nw.col("a").mean(), 0),
        (nw.col("a").cum_sum(), 1),
        (nw.col("a").cum_sum().over(_order_by="id"), 0),
        ((nw.col("a").cum_sum() + 1).over(_order_by="id"), 1),
        (nw.col("a").cum_sum().cum_sum().over(_order_by="id"), 1),
        (nw.col("a").cum_sum().cum_sum(), 2),
        (nw.sum_horizontal(nw.col("a"), nw.col("a").cum_sum()), 1),
        (nw.sum_horizontal(nw.col("a"), nw.col("a").cum_sum()).over("a"), 1),
    ],
)
def test_has_open_windows(expr: nw.Expr, expected: int) -> None:
    assert expr._metadata.n_open_windows == expected
