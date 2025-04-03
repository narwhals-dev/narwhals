from __future__ import annotations

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nw.col("a"), 0),
        (nw.col("a").mean(), 0),
        (nw.col("a").cum_sum(), 1),
        (nw.col("a").cum_sum().over(order_by="id"), 0),
        ((nw.col("a").cum_sum() + 1).over(order_by="id"), 1),
        (nw.col("a").cum_sum().cum_sum().over(order_by="id"), 1),
        (nw.col("a").cum_sum().cum_sum(), 2),
        (nw.sum_horizontal(nw.col("a"), nw.col("a").cum_sum()), 1),
        (nw.sum_horizontal(nw.col("a"), nw.col("a").cum_sum()).over("a"), 1),
    ],
)
def test_has_open_windows(expr: nw.Expr, expected: int) -> None:
    assert expr._metadata.n_open_windows == expected


def test_misleading_order_by() -> None:
    with pytest.raises(InvalidOperationError):
        nw.col("a").mean().over(order_by="b")
    with pytest.raises(InvalidOperationError):
        nw.col("a").rank().over(order_by="b")
