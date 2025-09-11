from __future__ import annotations

import operator
import re
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable

import pytest

import narwhals as nw
import narwhals._plan.functions as nwd
from narwhals._plan.common import ExprIR, Function
from narwhals._plan.dummy import Expr, Series
from narwhals._plan.expr_parsing import parse_into_seq_of_expr_ir
from narwhals._plan.expressions import boolean, expr, functions as F, operators as ops
from narwhals._plan.expressions.expr import BinaryExpr, FunctionExpr, RangeExpr
from narwhals._plan.expressions.literal import SeriesLiteral
from narwhals.exceptions import (
    InvalidIntoExprError,
    InvalidOperationError,
    InvalidOperationError as LengthChangingExprError,
    MultiOutputExpressionError,
    ShapeError,
)
from tests.plan.utils import assert_expr_ir_equal

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from typing_extensions import TypeAlias

    from narwhals._plan.typing import IntoExpr, IntoExprColumn, OperatorFn, Seq


IntoIterable: TypeAlias = Callable[[Sequence[Any]], Iterable[Any]]


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
        isinstance(node, ExprIR)
        for node in parse_into_seq_of_expr_ir(*exprs, **named_exprs)
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
        ([nwd.lit(1), nwd.lit(2, nw.Int64), nwd.lit(3, nw.Int64())]),
    ],
)
def test_function_expr_horizontal(
    function: Callable[..., Expr],
    ir_node: type[Function],
    args: Seq[IntoExpr | Iterable[IntoExpr]],
) -> None:
    variadic = function(*args)
    sequence = function(args)
    assert isinstance(variadic, Expr)
    assert isinstance(sequence, Expr)
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


def test_invalid_repeat_agg() -> None:
    with pytest.raises(InvalidOperationError):
        nwd.col("a").mean().mean()
    with pytest.raises(InvalidOperationError):
        nwd.col("a").first().max()
    with pytest.raises(InvalidOperationError):
        nwd.col("a").any().std()
    with pytest.raises(InvalidOperationError):
        nwd.col("a").all().quantile(0.5, "linear")
    with pytest.raises(InvalidOperationError):
        nwd.col("a").arg_max().min()
    with pytest.raises(InvalidOperationError):
        nwd.col("a").arg_min().arg_max()


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


def test_agg_non_elementwise_range_special() -> None:
    e = nwd.int_range(0, 100)
    assert isinstance(e._ir, RangeExpr)
    e = nwd.int_range(nwd.len(), dtype=nw.UInt32).alias("index")
    ir = e._ir
    assert isinstance(ir, expr.Alias)
    assert isinstance(ir.expr, RangeExpr)
    assert isinstance(ir.expr.input[0], expr.Literal)
    assert isinstance(ir.expr.input[1], expr.Len)


def test_invalid_int_range() -> None:
    pattern = re.compile(r"scalar.+agg", re.IGNORECASE)
    with pytest.raises(InvalidOperationError, match=pattern):
        nwd.int_range(nwd.col("a"))
    with pytest.raises(InvalidOperationError, match=pattern):
        nwd.int_range(nwd.nth(1), 10)
    with pytest.raises(InvalidOperationError, match=pattern):
        nwd.int_range(0, nwd.col("a").abs())
    with pytest.raises(InvalidOperationError, match=pattern):
        nwd.int_range(nwd.col("a") + 1)


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


def test_invalid_binary_expr_multi() -> None:
    pattern = re.escape("all() + cols(['b', 'c'])\n        ^^^^^^^^^^^^^^^^")
    with pytest.raises(MultiOutputExpressionError, match=pattern):
        nwd.all() + nwd.col("b", "c")
    pattern = re.escape(
        "index_columns((1, 2, 3)) * index_columns((4, 5, 6)).max()\n"
        "                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
    )
    with pytest.raises(MultiOutputExpressionError, match=pattern):
        nwd.nth(1, 2, 3) * nwd.nth(4, 5, 6).max()
    pattern = re.escape(
        "cols(['a', 'b', 'c']).abs().fill_null([lit(int: 0)]).round() * index_columns((9, 10)).cast(Int64).sort(asc)\n"
        "                                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
    )
    with pytest.raises(MultiOutputExpressionError, match=pattern):
        nwd.col("a", "b", "c").abs().fill_null(0).round(2) * nwd.nth(9, 10).cast(
            nw.Int64()
        ).sort()


def test_invalid_binary_expr_length_changing() -> None:
    a = nwd.col("a")
    b = nwd.col("b")

    with pytest.raises(LengthChangingExprError):
        a.unique() + b.unique()

    with pytest.raises(LengthChangingExprError):
        a.mode() * b.unique()

    with pytest.raises(LengthChangingExprError):
        a.drop_nulls() - b.mode()

    with pytest.raises(LengthChangingExprError):
        a.gather_every(2, 1) / b.drop_nulls()

    with pytest.raises(LengthChangingExprError):
        a.map_batches(lambda x: x) / b.gather_every(1, 0)


def _is_expr_ir_binary_expr(expr: Expr) -> bool:
    return isinstance(expr._ir, BinaryExpr)


def test_binary_expr_length_changing_agg() -> None:
    a = nwd.col("a")
    b = nwd.col("b")

    assert _is_expr_ir_binary_expr(a.unique().first() + b.unique())
    assert _is_expr_ir_binary_expr(a.mode().last() * b.unique())
    assert _is_expr_ir_binary_expr(a.drop_nulls().min() - b.mode())
    assert _is_expr_ir_binary_expr(a.gather_every(2, 1) / b.drop_nulls().max())
    assert _is_expr_ir_binary_expr(
        b.gather_every(1, 0)
        / a.map_batches(lambda x: x, returns_scalar=True, return_dtype=nw.Float64)
    )
    assert _is_expr_ir_binary_expr(
        b.unique() * a.map_batches(lambda x: x, return_dtype=nw.Unknown).first()
    )


def test_invalid_binary_expr_shape() -> None:
    pattern = re.compile(
        re.escape("Cannot combine length-changing expressions with length-preserving"),
        re.IGNORECASE,
    )
    a = nwd.col("a")
    b = nwd.col("b")

    with pytest.raises(ShapeError, match=pattern):
        a.unique() + b
    with pytest.raises(ShapeError, match=pattern):
        a.map_batches(lambda x: x, is_elementwise=True) * b.gather_every(1, 0)
    with pytest.raises(ShapeError, match=pattern):
        a / b.drop_nulls()


@pytest.mark.parametrize("into_iter", [list, tuple, deque, iter, dict.fromkeys, set])
def test_is_in_seq(into_iter: IntoIterable) -> None:
    expected = 1, 2, 3
    other = into_iter(list(expected))
    expr = nwd.col("a").is_in(other)
    ir = expr._ir
    assert isinstance(ir, FunctionExpr)
    assert isinstance(ir.function, boolean.IsInSeq)
    assert ir.function.other == expected


def test_is_in_series() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    native = pa.chunked_array([pa.array([1, 2, 3])])
    other = Series.from_native(native)
    expr = nwd.col("a").is_in(other)
    ir = expr._ir
    assert isinstance(ir, FunctionExpr)
    assert isinstance(ir.function, boolean.IsInSeries)
    assert ir.function.other.unwrap().to_native() is native


@pytest.mark.parametrize(
    ("other", "context"),
    [
        ("words", pytest.raises(TypeError, match=r"str \| bytes.+str")),
        (b"words", pytest.raises(TypeError, match=r"str \| bytes.+bytes")),
        (
            nwd.col("b"),
            pytest.raises(
                NotImplementedError, match=re.compile(r"iterable instead", re.IGNORECASE)
            ),
        ),
        (
            999,
            pytest.raises(
                TypeError, match=re.compile(r"only.+iterable.+int", re.IGNORECASE)
            ),
        ),
    ],
)
def test_invalid_is_in(other: Any, context: AbstractContextManager[Any]) -> None:
    with context:
        nwd.col("a").is_in(other)


def test_filter_full_spellings() -> None:
    a = nwd.col("a")
    b = nwd.col("b")
    c = nwd.col("c")
    d = nwd.col("d")
    expected = a.filter(b != b.max(), c < nwd.lit(2), d == nwd.lit(5))
    expr_1 = a.filter([b != b.max(), c < nwd.lit(2), d == nwd.lit(5)])
    expr_2 = a.filter([b != b.max(), c < nwd.lit(2)], d=nwd.lit(5))
    expr_3 = a.filter([b != b.max(), c < nwd.lit(2)], d=5)
    expr_4 = a.filter(b != b.max(), c < nwd.lit(2), d=5)
    expr_5 = a.filter(b != b.max(), c < 2, d=5)
    expr_6 = a.filter((b != b.max(), c < 2), d=5)
    assert_expr_ir_equal(expected, expr_1)
    assert_expr_ir_equal(expected, expr_2)
    assert_expr_ir_equal(expected, expr_3)
    assert_expr_ir_equal(expected, expr_4)
    assert_expr_ir_equal(expected, expr_5)
    assert_expr_ir_equal(expected, expr_6)


@pytest.mark.parametrize(
    ("predicates", "constraints", "context"),
    [
        ([nwd.col("b").is_last_distinct()], {}, nullcontext()),
        ((), {"b": 10}, nullcontext()),
        ((), {"b": nwd.lit(10)}, nullcontext()),
        (
            (),
            {},
            pytest.raises(
                TypeError, match=re.compile(r"at least one predicate", re.IGNORECASE)
            ),
        ),
        ((nwd.col("b") > 1, nwd.col("c").is_null()), {}, nullcontext()),
        (
            ([nwd.col("b") > 1], nwd.col("c").is_null()),
            {},
            pytest.raises(
                InvalidIntoExprError,
                match=re.compile(
                    r"both iterable.+positional.+not supported", re.IGNORECASE
                ),
            ),
        ),
    ],
)
def test_filter_partial_spellings(
    predicates: Iterable[IntoExprColumn],
    constraints: dict[str, Any],
    context: AbstractContextManager[Any],
) -> None:
    with context:
        assert nwd.col("a").filter(*predicates, **constraints)


def test_lit_series_roundtrip() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    data = ["a", "b", "c"]
    native = pa.chunked_array([pa.array(data)])
    series = Series.from_native(native)
    lit_series = nwd.lit(series)
    assert lit_series.meta.is_literal()
    ir = lit_series._ir
    assert isinstance(ir, expr.Literal)
    assert isinstance(ir.dtype, nw.String)
    assert isinstance(ir.value, SeriesLiteral)
    unwrapped = ir.unwrap()
    assert isinstance(unwrapped, Series)
    assert isinstance(unwrapped.to_native(), pa.ChunkedArray)
    assert unwrapped.to_list() == data


@pytest.mark.parametrize(
    ("arg_1", "arg_2", "function", "op"),
    [
        (nwd.col("a"), 1, operator.eq, ops.Eq),
        (nwd.col("a"), "b", operator.eq, ops.Eq),
        (nwd.col("a"), 1, operator.ne, ops.NotEq),
        (nwd.col("a"), "b", operator.ne, ops.NotEq),
        (nwd.col("a"), "b", operator.ge, ops.GtEq),
        (nwd.col("a"), "b", operator.gt, ops.Gt),
        (nwd.col("a"), "b", operator.le, ops.LtEq),
        (nwd.col("a"), "b", operator.lt, ops.Lt),
        ((nwd.col("a") != 1), False, operator.and_, ops.And),
        ((nwd.col("a") != 1), False, operator.or_, ops.Or),
        ((nwd.col("a")), True, operator.xor, ops.ExclusiveOr),
        (nwd.col("a"), 6, operator.add, ops.Add),
        (nwd.col("a"), 2.1, operator.mul, ops.Multiply),
        (nwd.col("a"), nwd.col("b"), operator.sub, ops.Sub),
        (nwd.col("a"), 2, operator.pow, F.Pow),
        (nwd.col("a"), 2, operator.mod, ops.Modulus),
        (nwd.col("a"), 2, operator.floordiv, ops.FloorDivide),
        (nwd.col("a"), 4, operator.truediv, ops.TrueDivide),
    ],
)
def test_operators_left_right(
    arg_1: IntoExpr,
    arg_2: IntoExpr,
    function: OperatorFn,
    op: type[ops.Operator | Function],
) -> None:
    inverse: Mapping[type[ops.Operator], type[ops.Operator]] = {
        ops.Gt: ops.Lt,
        ops.Lt: ops.Gt,
        ops.GtEq: ops.LtEq,
        ops.LtEq: ops.GtEq,
    }
    result_1 = function(arg_1, arg_2)
    result_2 = function(arg_2, arg_1)
    assert isinstance(result_1, Expr)
    assert isinstance(result_2, Expr)
    ir_1 = result_1._ir
    ir_2 = result_2._ir
    if op in {ops.Eq, ops.NotEq}:
        assert ir_1 == ir_2
    else:
        assert ir_1 != ir_2
    if issubclass(op, ops.Operator):
        assert isinstance(ir_1, BinaryExpr)
        assert isinstance(ir_1.op, op)
        assert isinstance(ir_2, BinaryExpr)
        op_inverse = inverse.get(op, op)
        assert isinstance(ir_2.op, op_inverse)
        if op in {ops.Eq, ops.NotEq, *inverse}:
            assert ir_1.left == ir_2.left
            assert ir_1.right == ir_2.right
        else:
            assert ir_1.left == ir_2.right
            assert ir_1.right == ir_2.left
    else:
        assert isinstance(ir_1, FunctionExpr)
        assert isinstance(ir_1.function, op)
        assert isinstance(ir_2, FunctionExpr)
        assert isinstance(ir_2.function, op)
        assert tuple(reversed(ir_2.input)) == ir_1.input
