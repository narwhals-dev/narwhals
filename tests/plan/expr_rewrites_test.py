from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals._plan import demo as nwd, expr_parsing as parse
from narwhals._plan.expr import WindowExpr
from narwhals._plan.expr_rewrites import rewrite_all, rewrite_elementwise_over
from narwhals._plan.window import Over
from narwhals.exceptions import InvalidOperationError
from tests.plan.utils import assert_expr_ir_equal

if TYPE_CHECKING:
    from narwhals._plan.typing import IntoExpr
    from narwhals.dtypes import DType


@pytest.fixture
def schema_2() -> dict[str, DType]:
    return {
        "a": nw.Int64(),
        "b": nw.Int64(),
        "c": nw.Int64(),
        "d": nw.Int64(),
        "e": nw.Int64(),
        "f": nw.String(),
        "g": nw.String(),
        "h": nw.String(),
        "i": nw.Boolean(),
        "j": nw.Boolean(),
        "k": nw.Boolean(),
    }


def _to_window_expr(into_expr: IntoExpr, *partition_by: IntoExpr) -> WindowExpr:
    return WindowExpr(
        expr=parse.parse_into_expr_ir(into_expr),
        partition_by=parse.parse_into_seq_of_expr_ir(*partition_by),
        options=Over(),
    )


def test_rewrite_elementwise_over_simple(schema_2: dict[str, DType]) -> None:
    with pytest.raises(InvalidOperationError, match=r"over.+elementwise"):
        nwd.col("a").sum().abs().over("b")

    # NOTE: Since the requested "before" expression is currently an error (at definition time),
    # we need to manually build the IR - to sidestep the validation in `Over.to_window_expr`.
    # Later, that error might not be needed if we can do this rewrite.
    # If you're here because of a "Did not raise" - just replace everything with the (previously) erroring expr.
    expected = nwd.col("a").sum().over("b").abs()
    before = _to_window_expr(nwd.col("a").sum().abs(), "b").to_narwhals()
    assert_expr_ir_equal(before, "col('a').sum().abs().over([col('b')])")
    actual = rewrite_all(before, schema=schema_2, rewrites=[rewrite_elementwise_over])
    assert len(actual) == 1
    assert_expr_ir_equal(actual[0], expected)


def test_rewrite_elementwise_over_multiple(schema_2: dict[str, DType]) -> None:
    expected = (
        nwd.col("b").last().over("d").replace_strict({1: 2}),
        nwd.col("c").last().over("d").replace_strict({1: 2}),
    )
    before = _to_window_expr(
        nwd.col("b", "c").last().replace_strict({1: 2}), "d"
    ).to_narwhals()
    assert_expr_ir_equal(
        before, "cols(['b', 'c']).last().replace_strict().over([col('d')])"
    )
    actual = rewrite_all(before, schema=schema_2, rewrites=[rewrite_elementwise_over])
    assert len(actual) == 2
    for lhs, rhs in zip(actual, expected):
        assert_expr_ir_equal(lhs, rhs)
