from __future__ import annotations

import pytest

import narwhals as nw
from narwhals._expression_parsing import WindowKind
from narwhals.exceptions import InvalidOperationError


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nw.col("a"), WindowKind.NONE),
        (nw.col("a").mean(), WindowKind.NONE),
        (nw.col("a").cum_sum(), WindowKind.CLOSEABLE),
        (nw.col("a").cum_sum().over(order_by="id"), WindowKind.CLOSED),
        (nw.col("a").cum_sum().abs().over(order_by="id"), WindowKind.UNCLOSEABLE),
        ((nw.col("a").cum_sum() + 1).over(order_by="id"), WindowKind.UNCLOSEABLE),
        (nw.col("a").cum_sum().cum_sum().over(order_by="id"), WindowKind.UNCLOSEABLE),
        (nw.col("a").cum_sum().cum_sum(), WindowKind.UNCLOSEABLE),
        (nw.sum_horizontal(nw.col("a"), nw.col("a").cum_sum()), WindowKind.UNCLOSEABLE),
        (
            nw.sum_horizontal(nw.col("a"), nw.col("a").cum_sum()).over("a"),
            WindowKind.UNCLOSEABLE,
        ),
        (
            nw.sum_horizontal(nw.col("a"), nw.col("a").cum_sum().over(order_by="i")),
            WindowKind.CLOSED,
        ),
        (
            nw.sum_horizontal(
                nw.col("a").diff(), nw.col("a").cum_sum().over(order_by="i")
            ),
            WindowKind.UNCLOSEABLE,
        ),
        (
            nw.sum_horizontal(nw.col("a").diff(), nw.col("a").cum_sum()).over(
                order_by="i"
            ),
            WindowKind.UNCLOSEABLE,
        ),
        (
            nw.sum_horizontal(nw.col("a").diff().abs(), nw.col("a").cum_sum()).over(
                order_by="i"
            ),
            WindowKind.UNCLOSEABLE,
        ),
    ],
)
def test_window_kind(expr: nw.Expr, expected: WindowKind) -> None:
    assert expr._metadata.window_kind is expected


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
