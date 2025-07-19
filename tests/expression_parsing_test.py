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
        (nw.col("a").cum_sum().abs().over(order_by="id"), 1),
        ((nw.col("a").cum_sum() + 1).over(order_by="id"), 1),
        (nw.col("a").cum_sum().cum_sum().over(order_by="id"), 1),
        (nw.col("a").cum_sum().cum_sum(), 2),
        (nw.sum_horizontal(nw.col("a"), nw.col("a").cum_sum()), 1),
        (nw.sum_horizontal(nw.col("a"), nw.col("a").cum_sum()).over(order_by="a"), 1),
        (nw.sum_horizontal(nw.col("a"), nw.col("a").cum_sum().over(order_by="i")), 0),
        (
            nw.sum_horizontal(
                nw.col("a").diff(), nw.col("a").cum_sum().over(order_by="i")
            ),
            1,
        ),
        (
            nw.sum_horizontal(nw.col("a").diff(), nw.col("a").cum_sum()).over(
                order_by="i"
            ),
            2,
        ),
        (
            nw.sum_horizontal(nw.col("a").diff().abs(), nw.col("a").cum_sum()).over(
                order_by="i"
            ),
            2,
        ),
    ],
)
def test_window_kind(expr: nw.Expr, expected: int) -> None:
    assert expr._metadata.n_orderable_ops == expected


def test_misleading_order_by() -> None:
    with pytest.raises(InvalidOperationError):
        nw.col("a").mean().over(order_by="b")


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


def test_rank_aggregation() -> None:
    with pytest.raises(InvalidOperationError):
        nw.col("a").mean().rank()
    with pytest.raises(InvalidOperationError):
        nw.col("a").mean().is_unique()


def test_diff_aggregation() -> None:
    with pytest.raises(InvalidOperationError):
        nw.col("a").mean().diff()


def test_invalid_over() -> None:
    with pytest.raises(InvalidOperationError):
        nw.col("a").fill_null(3).over("b")


def test_nested_over() -> None:
    with pytest.raises(InvalidOperationError):
        nw.col("a").mean().over("b").over("c")
    with pytest.raises(InvalidOperationError):
        nw.col("a").mean().over("b").over("c", order_by="i")


def test_filtration_over() -> None:
    with pytest.raises(InvalidOperationError):
        nw.col("a").drop_nulls().over("b")
    with pytest.raises(InvalidOperationError):
        nw.col("a").drop_nulls().over("b", order_by="i")
    with pytest.raises(InvalidOperationError):
        nw.col("a").diff().drop_nulls().over("b", order_by="i")
