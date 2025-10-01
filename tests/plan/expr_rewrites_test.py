from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals import _plan as nwp
from narwhals._plan import _parse, expressions as ir, selectors as ndcs
from narwhals._plan._rewrites import (
    rewrite_all,
    rewrite_binary_agg_over,
    rewrite_elementwise_over,
)
from narwhals._plan.expressions.window import Over
from narwhals.exceptions import InvalidOperationError
from tests.plan.utils import assert_expr_ir_equal, named_ir

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


def _to_window_expr(into_expr: IntoExpr, *partition_by: IntoExpr) -> ir.WindowExpr:
    return ir.WindowExpr(
        expr=_parse.parse_into_expr_ir(into_expr),
        partition_by=_parse.parse_into_seq_of_expr_ir(*partition_by),
        options=Over(),
    )


def test_rewrite_elementwise_over_simple(schema_2: dict[str, DType]) -> None:
    with pytest.raises(InvalidOperationError, match=r"over.+elementwise"):
        nwp.col("a").sum().abs().over("b")

    # NOTE: Since the requested "before" expression is currently an error (at definition time),
    # we need to manually build the IR - to sidestep the validation in `Over.to_window_expr`.
    # Later, that error might not be needed if we can do this rewrite.
    # If you're here because of a "Did not raise" - just replace everything with the (previously) erroring expr.
    expected = nwp.col("a").sum().over("b").abs()
    before = _to_window_expr(nwp.col("a").sum().abs(), "b").to_narwhals()
    assert_expr_ir_equal(before, "col('a').sum().abs().over([col('b')])")
    actual = rewrite_all(before, schema=schema_2, rewrites=[rewrite_elementwise_over])
    assert len(actual) == 1
    assert_expr_ir_equal(actual[0], expected)


def test_rewrite_elementwise_over_multiple(schema_2: dict[str, DType]) -> None:
    expected = (
        nwp.col("b").last().over("d").replace_strict({1: 2}),
        nwp.col("c").last().over("d").replace_strict({1: 2}),
    )
    before = _to_window_expr(
        nwp.col("b", "c").last().replace_strict({1: 2}), "d"
    ).to_narwhals()
    assert_expr_ir_equal(
        before, "cols(['b', 'c']).last().replace_strict().over([col('d')])"
    )
    actual = rewrite_all(before, schema=schema_2, rewrites=[rewrite_elementwise_over])
    assert len(actual) == 2
    for lhs, rhs in zip(actual, expected):
        assert_expr_ir_equal(lhs, rhs)


def test_rewrite_elementwise_over_complex(schema_2: dict[str, DType]) -> None:
    expected = (
        named_ir("a", nwp.col("a")),
        named_ir("b", nwp.col("b").cast(nw.String)),
        named_ir("x2", nwp.col("c").max().over("a").fill_null(50)),
        named_ir("d**", ~nwp.col("d").is_duplicated().over("b")),
        named_ir("f_some", nwp.col("f").str.contains("some")),
        named_ir("g_some", nwp.col("g").str.contains("some")),
        named_ir("h_some", nwp.col("h").str.contains("some")),
        named_ir("D", nwp.col("d").null_count().over("f", "g", "j").sqrt()),
        named_ir("E", nwp.col("e").null_count().over("f", "g", "j").sqrt()),
        named_ir("B", nwp.col("b").null_count().over("f", "g", "j").sqrt()),
    )
    before = (
        nwp.col("a"),
        nwp.col("b").cast(nw.String),
        (
            _to_window_expr(nwp.col("c").max().alias("x").fill_null(50), "a")
            .to_narwhals()
            .alias("x2")
        ),
        ~(nwp.col("d").is_duplicated().alias("d*")).alias("d**").over("b"),
        ndcs.string().str.contains("some").name.suffix("_some"),
        (
            _to_window_expr(nwp.nth(3, 4, 1).null_count().sqrt(), "f", "g", "j")
            .to_narwhals()
            .name.to_uppercase()
        ),
    )
    actual = rewrite_all(*before, schema=schema_2, rewrites=[rewrite_elementwise_over])
    assert len(actual) == len(expected)
    for lhs, rhs in zip(actual, expected):
        assert_expr_ir_equal(lhs, rhs)


def test_rewrite_binary_agg_over_simple(schema_2: dict[str, DType]) -> None:
    expected = (
        nwp.col("a") - nwp.col("a").mean().over("b"),
        nwp.col("c") * nwp.col("c").abs().null_count().over("d"),
    )
    before = (
        (nwp.col("a") - nwp.col("a").mean()).over("b"),
        (nwp.col("c") * nwp.col("c").abs().null_count()).over("d"),
    )
    actual = rewrite_all(*before, schema=schema_2, rewrites=[rewrite_binary_agg_over])
    assert len(actual) == 2
    for lhs, rhs in zip(actual, expected):
        assert_expr_ir_equal(lhs, rhs)


def test_rewrite_binary_agg_over_multiple(schema_2: dict[str, DType]) -> None:
    expected = (
        named_ir("hi_a", nwp.col("a") / nwp.col("e").drop_nulls().first().over("g")),
        named_ir("hi_b", nwp.col("b") / nwp.col("e").drop_nulls().first().over("g")),
        named_ir("hi_c", nwp.col("c") / nwp.col("e").drop_nulls().first().over("g")),
        named_ir("hi_d", nwp.col("d") / nwp.col("e").drop_nulls().first().over("g")),
    )
    before = (
        (nwp.col("a", "b", "c", "d") / nwp.col("e").drop_nulls().first()).over("g")
    ).name.prefix("hi_")
    actual = rewrite_all(before, schema=schema_2, rewrites=[rewrite_binary_agg_over])
    assert len(actual) == 4
    for lhs, rhs in zip(actual, expected):
        assert_expr_ir_equal(lhs, rhs)
