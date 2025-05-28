from __future__ import annotations

import re
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


def test_valid_windows() -> None:
    """Was planning to test this matched, but we seem to allow elementwise horizontal?

    https://github.com/narwhals-dev/narwhals/blob/63c8e4771a1df4e0bfeea5559c303a4a447d5cc2/tests/expression_parsing_test.py#L10-L45
    """
    ELEMENTWISE_ERR = re.compile(r"cannot use.+over.+elementwise", re.IGNORECASE)  # noqa: N806
    a = nwd.col("a")
    assert a.cum_sum()
    assert a.cum_sum().over(order_by="id")
    with pytest.raises(InvalidOperationError, match=ELEMENTWISE_ERR):
        assert a.cum_sum().abs().over(order_by="id")

    assert (a.cum_sum() + 1).over(order_by="id")
    assert a.cum_sum().cum_sum().over(order_by="id")
    assert a.cum_sum().cum_sum()
    assert nwd.sum_horizontal(a, a.cum_sum())
    with pytest.raises(InvalidOperationError, match=ELEMENTWISE_ERR):
        assert nwd.sum_horizontal(a, a.cum_sum()).over(order_by="a")

    assert nwd.sum_horizontal(a, a.cum_sum().over(order_by="i"))
    assert nwd.sum_horizontal(a.diff(), a.cum_sum().over(order_by="i"))
    with pytest.raises(InvalidOperationError, match=ELEMENTWISE_ERR):
        assert nwd.sum_horizontal(a.diff(), a.cum_sum()).over(order_by="i")

    with pytest.raises(InvalidOperationError, match=ELEMENTWISE_ERR):
        assert nwd.sum_horizontal(a.diff().abs(), a.cum_sum()).over(order_by="i")


# TODO @dangotbanned: Get parity with the existing tests
# https://github.com/narwhals-dev/narwhals/blob/63c8e4771a1df4e0bfeea5559c303a4a447d5cc2/tests/expression_parsing_test.py#L48-L105


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


# TODO @dangotbanned: Add `head`, `tail`
# head/tail are implemented in terms of `Expr::Slice`
# We don't support `Expr.slice`, seems odd to add it for a deprecation 🤔
# polars allows this in `select`, but not `with_columns`
def test_head_aggregation() -> None:
    with pytest.raises(InvalidOperationError):
        nwd.col("a").mean().head()  # type: ignore[attr-defined]


# TODO @dangotbanned: Non-`polars`` rule
def test_misleading_order_by() -> None:
    with pytest.raises(InvalidOperationError):
        nwd.col("a").mean().over(order_by="b")
    with pytest.raises(InvalidOperationError):
        nwd.col("a").rank().over(order_by="b")


# NOTE: Previously multiple different errors, but they can be reduced to the same thing
# Once we are scalar, only elementwise is allowed
def test_invalid_agg_non_elementwise() -> None:
    pattern = re.compile(r"cannot use.+rank.+aggregated.+mean", re.IGNORECASE)
    with pytest.raises(InvalidOperationError, match=pattern):
        nwd.col("a").mean().rank()
    pattern = re.compile(r"cannot use.+drop_nulls.+aggregated.+max", re.IGNORECASE)
    with pytest.raises(InvalidOperationError):
        nwd.col("a").max().drop_nulls()
    pattern = re.compile(r"cannot use.+diff.+aggregated.+min", re.IGNORECASE)
    with pytest.raises(InvalidOperationError):
        nwd.col("a").min().diff()


# NOTE: Non-`polars`` rule
def test_invalid_over() -> None:
    pattern = re.compile(r"cannot use.+over.+elementwise", re.IGNORECASE)
    with pytest.raises(InvalidOperationError, match=pattern):
        nwd.col("a").fill_null(3).over("b")


def test_nested_over() -> None:
    pattern = re.compile(r"cannot nest.+over", re.IGNORECASE)
    with pytest.raises(InvalidOperationError, match=pattern):
        nwd.col("a").mean().over("b").over("c")
    with pytest.raises(InvalidOperationError, match=pattern):
        nwd.col("a").mean().over("b").over("c", order_by="i")


# NOTE: This *can* error in polars, but only if the length **actually changes**
# The rule then breaks down to needing the same length arrays in all parts of the over
def test_filtration_over() -> None:
    pattern = re.compile(r"cannot use.+over.+change length", re.IGNORECASE)
    with pytest.raises(InvalidOperationError, match=pattern):
        nwd.col("a").drop_nulls().over("b")
    with pytest.raises(InvalidOperationError, match=pattern):
        nwd.col("a").drop_nulls().over("b", order_by="i")
    with pytest.raises(InvalidOperationError, match=pattern):
        nwd.col("a").diff().drop_nulls().over("b", order_by="i")
