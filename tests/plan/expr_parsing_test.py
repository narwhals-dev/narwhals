from __future__ import annotations

import operator
import re
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals import _plan as nwp
from narwhals._plan import expressions as ir
from narwhals._plan._parse import parse_into_seq_of_expr_ir
from narwhals._plan.expressions import functions as F, operators as ops
from narwhals._plan.expressions.literal import SeriesLiteral
from narwhals.exceptions import (
    ComputeError,
    InvalidIntoExprError,
    InvalidOperationError,
    InvalidOperationError as LengthChangingExprError,
    ShapeError,
)
from tests.plan.utils import assert_expr_ir_equal, re_compile

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from typing_extensions import TypeAlias

    from narwhals._plan._function import Function
    from narwhals._plan.typing import IntoExpr, IntoExprColumn, OperatorFn, Seq


IntoIterable: TypeAlias = Callable[[Sequence[Any]], Iterable[Any]]


@pytest.mark.parametrize(
    ("exprs", "named_exprs"),
    [
        ([nwp.col("a")], {}),
        (["a"], {}),
        ([], {"a": "b"}),
        ([], {"a": nwp.col("b")}),
        (["a", "b", nwp.col("c", "d", "e")], {"g": nwp.lit(1)}),
        ([["a", "b", "c"]], {"q": nwp.lit(5, nw.Int8())}),
        (
            [[nwp.nth(1), nwp.nth(2, 3, 4)]],
            {"n": nwp.col("p").count(), "other n": nwp.len()},
        ),
    ],
)
def test_parsing(
    exprs: Seq[IntoExpr | Iterable[IntoExpr]], named_exprs: dict[str, IntoExpr]
) -> None:
    assert all(
        isinstance(node, ir.ExprIR)
        for node in parse_into_seq_of_expr_ir(*exprs, **named_exprs)
    )


@pytest.mark.parametrize(
    ("function", "ir_node"),
    [
        (nwp.all_horizontal, ir.boolean.AllHorizontal),
        (nwp.any_horizontal, ir.boolean.AnyHorizontal),
        (nwp.sum_horizontal, F.SumHorizontal),
        (nwp.min_horizontal, F.MinHorizontal),
        (nwp.max_horizontal, F.MaxHorizontal),
        (nwp.mean_horizontal, F.MeanHorizontal),
    ],
)
@pytest.mark.parametrize(
    "args",
    [
        ("a", "b", "c"),
        (["a", "b", "c"]),
        (nwp.col("d", "e", "f"), nwp.col("g"), "q", nwp.nth(9)),
        ((nwp.lit(1),)),
        ([nwp.lit(1), nwp.lit(2, nw.Int64), nwp.lit(3, nw.Int64())]),
    ],
)
def test_function_expr_horizontal(
    function: Callable[..., nwp.Expr],
    ir_node: type[Function],
    args: Seq[IntoExpr | Iterable[IntoExpr]],
) -> None:
    variadic = function(*args)
    sequence = function(args)
    assert isinstance(variadic, nwp.Expr)
    assert isinstance(sequence, nwp.Expr)
    variadic_node = variadic._ir
    sequence_node = sequence._ir
    unrelated_node = nwp.lit(1)._ir
    assert isinstance(variadic_node, ir.FunctionExpr)
    assert isinstance(variadic_node.function, ir_node)
    assert variadic_node == sequence_node
    assert sequence_node != unrelated_node


def test_valid_windows() -> None:
    """Was planning to test this matched, but we seem to allow elementwise horizontal?

    https://github.com/narwhals-dev/narwhals/blob/63c8e4771a1df4e0bfeea5559c303a4a447d5cc2/tests/expression_parsing_test.py#L10-L45
    """
    ELEMENTWISE_ERR = re.compile(r"cannot use.+over.+elementwise", re.IGNORECASE)  # noqa: N806
    a = nwp.col("a")
    assert a.cum_sum()
    assert a.cum_sum().over(order_by="id")
    with pytest.raises(InvalidOperationError, match=ELEMENTWISE_ERR):
        assert a.cum_sum().abs().over(order_by="id")

    assert (a.cum_sum() + 1).over(order_by="id")
    assert a.cum_sum().cum_sum().over(order_by="id")
    assert a.cum_sum().cum_sum()
    assert nwp.sum_horizontal(a, a.cum_sum())
    with pytest.raises(InvalidOperationError, match=ELEMENTWISE_ERR):
        assert nwp.sum_horizontal(a, a.cum_sum()).over(order_by="a")

    assert nwp.sum_horizontal(a, a.cum_sum().over(order_by="i"))
    assert nwp.sum_horizontal(a.diff(), a.cum_sum().over(order_by="i"))
    with pytest.raises(InvalidOperationError, match=ELEMENTWISE_ERR):
        assert nwp.sum_horizontal(a.diff(), a.cum_sum()).over(order_by="i")

    with pytest.raises(InvalidOperationError, match=ELEMENTWISE_ERR):
        assert nwp.sum_horizontal(a.diff().abs(), a.cum_sum()).over(order_by="i")


def test_invalid_repeat_agg() -> None:
    with pytest.raises(InvalidOperationError):
        nwp.col("a").mean().mean()
    with pytest.raises(InvalidOperationError):
        nwp.col("a").first().max()
    with pytest.raises(InvalidOperationError):
        nwp.col("a").any().std()
    with pytest.raises(InvalidOperationError):
        nwp.col("a").all().quantile(0.5, "linear")
    with pytest.raises(InvalidOperationError):
        nwp.col("a").arg_max().min()
    with pytest.raises(InvalidOperationError):
        nwp.col("a").arg_min().arg_max()


# NOTE: Previously multiple different errors, but they can be reduced to the same thing
# Once we are scalar, only elementwise is allowed
def test_invalid_agg_non_elementwise() -> None:
    pattern = re.compile(r"cannot use.+rank.+aggregated.+mean", re.IGNORECASE)
    with pytest.raises(InvalidOperationError, match=pattern):
        nwp.col("a").mean().rank()
    pattern = re.compile(r"cannot use.+drop_nulls.+aggregated.+max", re.IGNORECASE)
    with pytest.raises(InvalidOperationError):
        nwp.col("a").max().drop_nulls()
    pattern = re.compile(r"cannot use.+diff.+aggregated.+min", re.IGNORECASE)
    with pytest.raises(InvalidOperationError):
        nwp.col("a").min().diff()


def test_agg_non_elementwise_range_special() -> None:
    e = nwp.int_range(0, 100)
    assert isinstance(e._ir, ir.RangeExpr)
    e = nwp.int_range(nwp.len(), dtype=nw.UInt32).alias("index")
    e_ir = e._ir
    assert isinstance(e_ir, ir.Alias)
    assert isinstance(e_ir.expr, ir.RangeExpr)
    assert isinstance(e_ir.expr.input[0], ir.Literal)
    assert isinstance(e_ir.expr.input[1], ir.Len)


def test_invalid_int_range() -> None:
    pattern = re.compile(r"scalar.+agg", re.IGNORECASE)
    with pytest.raises(InvalidOperationError, match=pattern):
        nwp.int_range(nwp.col("a"))
    with pytest.raises(InvalidOperationError, match=pattern):
        nwp.int_range(nwp.nth(1), 10)
    with pytest.raises(InvalidOperationError, match=pattern):
        nwp.int_range(0, nwp.col("a").abs())
    with pytest.raises(InvalidOperationError, match=pattern):
        nwp.int_range(nwp.col("a") + 1)


# NOTE: Non-`polars`` rule
def test_invalid_over() -> None:
    pattern = re.compile(r"cannot use.+over.+elementwise", re.IGNORECASE)
    with pytest.raises(InvalidOperationError, match=pattern):
        nwp.col("a").fill_null(3).over("b")


def test_nested_over() -> None:
    pattern = re.compile(r"cannot nest.+over", re.IGNORECASE)
    with pytest.raises(InvalidOperationError, match=pattern):
        nwp.col("a").mean().over("b").over("c")
    with pytest.raises(InvalidOperationError, match=pattern):
        nwp.col("a").mean().over("b").over("c", order_by="i")


# NOTE: This *can* error in polars, but only if the length **actually changes**
# The rule then breaks down to needing the same length arrays in all parts of the over
def test_filtration_over() -> None:
    pattern = re.compile(r"cannot use.+over.+change length", re.IGNORECASE)
    with pytest.raises(InvalidOperationError, match=pattern):
        nwp.col("a").drop_nulls().over("b")
    with pytest.raises(InvalidOperationError, match=pattern):
        nwp.col("a").drop_nulls().over("b", order_by="i")
    with pytest.raises(InvalidOperationError, match=pattern):
        nwp.col("a").diff().drop_nulls().over("b", order_by="i")


def test_invalid_binary_expr_length_changing() -> None:
    a = nwp.col("a")
    b = nwp.col("b")

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


def _is_expr_ir_binary_expr(expr: nwp.Expr) -> bool:
    return isinstance(expr._ir, ir.BinaryExpr)


def test_binary_expr_length_changing_agg() -> None:
    a = nwp.col("a")
    b = nwp.col("b")

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
    a = nwp.col("a")
    b = nwp.col("b")

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
    expr = nwp.col("a").is_in(other)
    e_ir = expr._ir
    assert isinstance(e_ir, ir.FunctionExpr)
    assert isinstance(e_ir.function, ir.boolean.IsInSeq)
    assert e_ir.function.other == expected


def test_is_in_series() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    native = pa.chunked_array([pa.array([1, 2, 3])])
    other = nwp.Series.from_native(native)
    expr = nwp.col("a").is_in(other)
    e_ir = expr._ir
    assert isinstance(e_ir, ir.FunctionExpr)
    assert isinstance(e_ir.function, ir.boolean.IsInSeries)
    assert e_ir.function.other.unwrap().to_native() is native


@pytest.mark.parametrize(
    ("other", "context"),
    [
        ("words", pytest.raises(TypeError, match=r"str \| bytes.+str")),
        (b"words", pytest.raises(TypeError, match=r"str \| bytes.+bytes")),
        (
            nwp.col("b"),
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
        nwp.col("a").is_in(other)


def test_filter_full_spellings() -> None:
    a = nwp.col("a")
    b = nwp.col("b")
    c = nwp.col("c")
    d = nwp.col("d")
    expected = a.filter(b != b.max(), c < nwp.lit(2), d == nwp.lit(5))
    expr_1 = a.filter([b != b.max(), c < nwp.lit(2), d == nwp.lit(5)])
    expr_2 = a.filter([b != b.max(), c < nwp.lit(2)], d=nwp.lit(5))
    expr_3 = a.filter([b != b.max(), c < nwp.lit(2)], d=5)
    expr_4 = a.filter(b != b.max(), c < nwp.lit(2), d=5)
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
        ([nwp.col("b").is_last_distinct()], {}, nullcontext()),
        ((), {"b": 10}, nullcontext()),
        ((), {"b": nwp.lit(10)}, nullcontext()),
        (
            (),
            {},
            pytest.raises(
                TypeError, match=re.compile(r"at least one predicate", re.IGNORECASE)
            ),
        ),
        ((nwp.col("b") > 1, nwp.col("c").is_null()), {}, nullcontext()),
        (
            ([nwp.col("b") > 1], nwp.col("c").is_null()),
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
        assert nwp.col("a").filter(*predicates, **constraints)


def test_lit_series_roundtrip() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    data = ["a", "b", "c"]
    native = pa.chunked_array([pa.array(data)])
    series = nwp.Series.from_native(native)
    lit_series = nwp.lit(series)
    assert lit_series.meta.is_literal()
    e_ir = lit_series._ir
    assert isinstance(e_ir, ir.Literal)
    assert isinstance(e_ir.dtype, nw.String)
    assert isinstance(e_ir.value, SeriesLiteral)
    unwrapped = e_ir.unwrap()
    assert isinstance(unwrapped, nwp.Series)
    assert isinstance(unwrapped.to_native(), pa.ChunkedArray)
    assert unwrapped.to_list() == data


@pytest.mark.parametrize(
    ("arg_1", "arg_2", "function", "op"),
    [
        (nwp.col("a"), 1, operator.eq, ops.Eq),
        (nwp.col("a"), "b", operator.eq, ops.Eq),
        (nwp.col("a"), 1, operator.ne, ops.NotEq),
        (nwp.col("a"), "b", operator.ne, ops.NotEq),
        (nwp.col("a"), "b", operator.ge, ops.GtEq),
        (nwp.col("a"), "b", operator.gt, ops.Gt),
        (nwp.col("a"), "b", operator.le, ops.LtEq),
        (nwp.col("a"), "b", operator.lt, ops.Lt),
        ((nwp.col("a") != 1), False, operator.and_, ops.And),
        ((nwp.col("a") != 1), False, operator.or_, ops.Or),
        ((nwp.col("a")), True, operator.xor, ops.ExclusiveOr),
        (nwp.col("a"), 6, operator.add, ops.Add),
        (nwp.col("a"), 2.1, operator.mul, ops.Multiply),
        (nwp.col("a"), nwp.col("b"), operator.sub, ops.Sub),
        (nwp.col("a"), 2, operator.pow, F.Pow),
        (nwp.col("a"), 2, operator.mod, ops.Modulus),
        (nwp.col("a"), 2, operator.floordiv, ops.FloorDivide),
        (nwp.col("a"), 4, operator.truediv, ops.TrueDivide),
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
    assert isinstance(result_1, nwp.Expr)
    assert isinstance(result_2, nwp.Expr)
    ir_1 = result_1._ir
    ir_2 = result_2._ir
    if op in {ops.Eq, ops.NotEq}:
        assert ir_1 == ir_2
    else:
        assert ir_1 != ir_2
    if issubclass(op, ops.Operator):
        assert isinstance(ir_1, ir.BinaryExpr)
        assert isinstance(ir_1.op, op)
        assert isinstance(ir_2, ir.BinaryExpr)
        op_inverse = inverse.get(op, op)
        assert isinstance(ir_2.op, op_inverse)
        if op in {ops.Eq, ops.NotEq, *inverse}:
            assert ir_1.left == ir_2.left
            assert ir_1.right == ir_2.right
        else:
            assert ir_1.left == ir_2.right
            assert ir_1.right == ir_2.left
    else:
        assert isinstance(ir_1, ir.FunctionExpr)
        assert isinstance(ir_1.function, op)
        assert isinstance(ir_2, ir.FunctionExpr)
        assert isinstance(ir_2.function, op)
        assert tuple(reversed(ir_2.input)) == ir_1.input


def test_hist_bins() -> None:
    bins_values = (0, 1.5, 3.0, 4.5, 6.0)
    a = nwp.col("a")
    hist_1 = a.hist(deque(bins_values), include_breakpoint=False)
    hist_2 = a.hist(list(bins_values), include_breakpoint=False)

    ir_1 = hist_1._ir
    ir_2 = hist_2._ir
    assert isinstance(ir_1, ir.FunctionExpr)
    assert isinstance(ir_2, ir.FunctionExpr)
    assert isinstance(ir_1.function, F.HistBins)
    assert isinstance(ir_2.function, F.HistBins)
    assert ir_1.function.include_breakpoint is False
    assert_expr_ir_equal(ir_1, ir_2)


def test_hist_bin_count() -> None:
    bin_count_default = 10
    include_breakpoint_default = True
    a = nwp.col("a")
    hist_1 = a.hist(
        bin_count=bin_count_default, include_breakpoint=include_breakpoint_default
    )
    hist_2 = a.hist()
    hist_3 = a.hist(bin_count=5)
    hist_4 = a.hist(include_breakpoint=False)

    ir_1 = hist_1._ir
    ir_2 = hist_2._ir
    ir_3 = hist_3._ir
    ir_4 = hist_4._ir
    assert isinstance(ir_1, ir.FunctionExpr)
    assert isinstance(ir_2, ir.FunctionExpr)
    assert isinstance(ir_3, ir.FunctionExpr)
    assert isinstance(ir_4, ir.FunctionExpr)
    assert isinstance(ir_1.function, F.HistBinCount)
    assert isinstance(ir_2.function, F.HistBinCount)
    assert isinstance(ir_3.function, F.HistBinCount)
    assert isinstance(ir_4.function, F.HistBinCount)
    assert ir_1.function.include_breakpoint is include_breakpoint_default
    assert ir_2.function.bin_count == bin_count_default
    assert_expr_ir_equal(ir_1, ir_2)
    assert ir_3.function.include_breakpoint != ir_4.function.include_breakpoint
    assert ir_4.function.bin_count != ir_3.function.bin_count
    assert ir_4 != ir_2
    assert ir_3 != ir_1


def test_hist_invalid() -> None:
    a = nwp.col("a")
    with pytest.raises(ComputeError, match=r"bin_count.+or.+bins"):
        a.hist(bins=[1], bin_count=1)
    with pytest.raises(ComputeError, match=r"bins.+monotonic"):
        a.hist([1, 5, 4])
    with pytest.raises(ComputeError, match=r"bins.+monotonic"):
        a.hist(deque((3, 2, 1)))
    with pytest.raises(TypeError):
        a.hist(1)  # type: ignore[arg-type]


def test_into_expr_invalid() -> None:
    pytest.importorskip("polars")
    import polars as pl

    with pytest.raises(
        TypeError, match=re_compile(r"expected.+narwhals.+got.+polars.+hint")
    ):
        nwp.col("a").max().over(pl.col("b"))  # type: ignore[arg-type]


def test_when_invalid() -> None:
    pattern = re_compile(r"multi-output expr.+not supported in.+when.+context")

    when = nwp.when(nwp.col("a", "b", "c").is_finite())
    when_then = when.then(nwp.col("d").is_unique())
    when_then_when = when_then.when(
        (nwp.median("a", "b", "c") > 2) | nwp.col("d").is_nan()
    )
    with pytest.raises(MultiOutputExpressionError, match=pattern):
        when.then(nwp.max("c", "d"))
    with pytest.raises(MultiOutputExpressionError, match=pattern):
        when_then.otherwise(nwp.min("h", "i", "j"))
    with pytest.raises(MultiOutputExpressionError, match=pattern):
        when_then_when.then(nwp.col(["b", "y", "e"]))


# NOTE: `Then`, `ChainedThen` use multi-inheritance, but **need** to use `Expr.__eq__`
def test_then_equal() -> None:
    expr = nwp.col("a").clip(nwp.col("a").kurtosis(), nwp.col("a").log())
    other = "other"
    then = nwp.when(a="b").then(nwp.col("c").skew())
    chained_then = then.when("d").then("e")

    assert isinstance(then == expr, nwp.Expr)
    assert isinstance(then == other, nwp.Expr)

    assert isinstance(chained_then == expr, nwp.Expr)
    assert isinstance(chained_then == other, nwp.Expr)

    assert isinstance(then == chained_then, nwp.Expr)
