from __future__ import annotations

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError


@pytest.mark.parametrize(
    ("expr", "expected_open", "expected_closed"),
    [
        (nw.col("a"), 0, False),
        (nw.col("a").mean(), 0, False),
        (nw.col("a").cum_sum(), 1, False),
        (nw.col("a").cum_sum().over(order_by="id"), 0, True),
        (nw.col("a").cum_sum().abs().over(order_by="id"), 1, True),
        ((nw.col("a").cum_sum() + 1).over(order_by="id"), 1, True),
        (nw.col("a").cum_sum().cum_sum().over(order_by="id"), 1, True),
        (nw.col("a").cum_sum().cum_sum(), 2, False),
        (nw.sum_horizontal(nw.col("a"), nw.col("a").cum_sum()), 1, False),
        (nw.sum_horizontal(nw.col("a"), nw.col("a").cum_sum()).over("a"), 1, True),
    ],
)
def test_has_open_windows(
    expr: nw.Expr, expected_open: int, *, expected_closed: bool
) -> None:
    assert expr._metadata.n_open_windows == expected_open
    assert expr._metadata.has_closed_windows is expected_closed


def test_misleading_order_by() -> None:
    with pytest.raises(InvalidOperationError):
        nw.col("a").mean().over(order_by="b")
    with pytest.raises(InvalidOperationError):
        nw.col("a").rank().over(order_by="b")


def test_double_over() -> None:
    with pytest.raises(InvalidOperationError):
        nw.col("a").mean().over("b").over("c")


def test_double_agg() -> None:
    with pytest.raises(InvalidOperationError):
        nw.col("a").mean().mean()
    with pytest.raises(InvalidOperationError):
        nw.col("a").mean().arg_max()


def test_filter_aggregation() -> None:
    with pytest.raises(InvalidOperationError):
        nw.col("a").mean().drop_nulls()
