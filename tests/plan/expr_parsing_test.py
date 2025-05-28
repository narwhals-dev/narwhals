from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable

import pytest

import narwhals as nw
import narwhals._plan.demo as nwd
from narwhals._plan import (
    boolean,
    functions as F,  # noqa: N812
)
from narwhals._plan.common import ExprIR, Function
from narwhals._plan.dummy import DummyExpr
from narwhals._plan.expr import FunctionExpr
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from narwhals._plan.common import IntoExpr, Seq


@pytest.mark.parametrize(
    ("exprs", "named_exprs"),
    [
        ([nwd.col("a")], {}),
        (["a"], {}),
        ([], {"a": "b"}),
        ([], {"a": nwd.col("b")}),
        (["a", "b", nwd.col("c", "d", "e")], {"g": nwd.lit(1)}),
        ([["a", "b", "c"]], {"q": nwd.lit(5, nw.Int8())}),
        (
            [[nwd.nth(1), nwd.nth(2, 3, 4)]],
            {"n": nwd.col("p").count(), "other n": nwd.len()},
        ),
    ],
)
def test_parsing(
    exprs: Seq[IntoExpr | Iterable[IntoExpr]], named_exprs: dict[str, IntoExpr]
) -> None:
    assert all(
        isinstance(node, ExprIR) for node in nwd.select_context(*exprs, **named_exprs)
    )


@pytest.mark.parametrize(
    ("function", "ir_node"),
    [
        (nwd.all_horizontal, boolean.AllHorizontal),
        (nwd.any_horizontal, boolean.AnyHorizontal),
        (nwd.sum_horizontal, F.SumHorizontal),
        (nwd.min_horizontal, F.MinHorizontal),
        (nwd.max_horizontal, F.MaxHorizontal),
        (nwd.mean_horizontal, F.MeanHorizontal),
    ],
)
@pytest.mark.parametrize(
    "args",
    [
        ("a", "b", "c"),
        (["a", "b", "c"]),
        (nwd.col("d", "e", "f"), nwd.col("g"), "q", nwd.nth(9)),
        ((nwd.lit(1),)),
        ([nwd.lit(1), nwd.lit(2), nwd.lit(3)]),
    ],
)
def test_function_expr_horizontal(
    function: Callable[..., DummyExpr],
    ir_node: type[Function],
    args: Seq[IntoExpr | Iterable[IntoExpr]],
) -> None:
    variadic = function(*args)
    sequence = function(args)
    assert isinstance(variadic, DummyExpr)
    assert isinstance(sequence, DummyExpr)
    variadic_node = variadic._ir
    sequence_node = sequence._ir
    unrelated_node = nwd.lit(1)._ir
    assert isinstance(variadic_node, FunctionExpr)
    assert isinstance(variadic_node.function, ir_node)
    assert variadic_node == sequence_node
    assert sequence_node != unrelated_node


# TODO @dangotbanned: Get partity with the existing tests
# https://github.com/narwhals-dev/narwhals/blob/63c8e4771a1df4e0bfeea5559c303a4a447d5cc2/tests/expression_parsing_test.py#L48-L105


def test_misleading_order_by() -> None:
    with pytest.raises(InvalidOperationError):
        nw.col("a").mean().over(order_by="b")
    with pytest.raises(InvalidOperationError):
        nw.col("a").rank().over(order_by="b")


# `test_double_over` is already covered in the later `test_nested_over`


# test_double_agg
def test_invalid_repeat_agg() -> None:
    with pytest.raises(InvalidOperationError):
        nwd.col("a").mean().mean()
    with pytest.raises(InvalidOperationError):
        nwd.col("a").first().max()
    with pytest.raises(InvalidOperationError):
        nwd.col("a").any().std()
    with pytest.raises(InvalidOperationError):
        nwd.col("a").all().quantile(0.5, "linear")


def test_filter_aggregation() -> None:
    with pytest.raises(InvalidOperationError):
        nwd.col("a").mean().drop_nulls()


# TODO @dangotbanned: Add `head`, `tail`
def test_head_aggregation() -> None:
    with pytest.raises(InvalidOperationError):
        nwd.col("a").mean().head()  # type: ignore[attr-defined]


def test_rank_aggregation() -> None:
    with pytest.raises(InvalidOperationError):
        nwd.col("a").mean().rank()


def test_diff_aggregation() -> None:
    with pytest.raises(InvalidOperationError):
        nwd.col("a").mean().diff()


def test_invalid_over() -> None:
    with pytest.raises(InvalidOperationError):
        nwd.col("a").fill_null(3).over("b")


def test_nested_over() -> None:
    with pytest.raises(InvalidOperationError):
        nwd.col("a").mean().over("b").over("c")
    with pytest.raises(InvalidOperationError):
        nwd.col("a").mean().over("b").over("c", order_by="i")


def test_filtration_over() -> None:
    with pytest.raises(InvalidOperationError):
        nwd.col("a").drop_nulls().over("b")
    with pytest.raises(InvalidOperationError):
        nwd.col("a").drop_nulls().over("b", order_by="i")
    with pytest.raises(InvalidOperationError):
        nwd.col("a").diff().drop_nulls().over("b", order_by="i")
