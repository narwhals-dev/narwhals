from __future__ import annotations

import datetime as dt
import operator
import re
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals import _plan as nwp
from narwhals._plan import expressions as ir, selectors as ncs
from narwhals._plan._parse import into_iter_expr_ir
from narwhals._plan.expressions import (
    aggregation as agg,
    functions as F,
    operators as ops,
)
from narwhals._plan.expressions.ranges import IntRange
from narwhals._utils import Implementation
from narwhals.dtypes import DType, Int64, List, Struct
from narwhals.exceptions import ComputeError, InvalidOperationError, ShapeError
from tests.plan.utils import assert_equal_data, assert_expr_ir_equal, re_compile

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from typing_extensions import TypeAlias

    from narwhals._plan._function import Function
    from narwhals._plan.typing import (
        IntoExpr,
        IntoExprColumn,
        OneOrIterable,
        OperatorFn,
        Seq,
    )
    from narwhals._plan.when_then import ChainedWhen, When
    from narwhals.typing import IntoDType, PythonLiteral


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
def test_parsing(exprs: Iterable[IntoExpr], named_exprs: dict[str, IntoExpr]) -> None:
    assert all(
        isinstance(node, ir.ExprIR) for node in into_iter_expr_ir(*exprs, **named_exprs)
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


def test_repeat_agg_invalid() -> None:
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
def test_agg_non_elementwise_invalid() -> None:
    pattern = re.compile(r"cannot use.+rank.+aggregated.+mean", re.IGNORECASE)
    with pytest.raises(InvalidOperationError, match=pattern):
        nwp.col("a").mean().rank()
    pattern = re.compile(r"cannot use.+drop_nulls.+aggregated.+max", re.IGNORECASE)
    with pytest.raises(InvalidOperationError):
        nwp.col("a").max().drop_nulls()
    pattern = re.compile(r"cannot use.+diff.+aggregated.+min", re.IGNORECASE)
    with pytest.raises(InvalidOperationError):
        nwp.col("a").min().diff()
    pattern = re.compile(r"cannot use.+map_batches.+aggregated.+first", re.IGNORECASE)
    with pytest.raises(InvalidOperationError):
        nwp.col("a").first().map_batches(lambda x: x, nw.Int64(), returns_scalar=True)


def test_agg_non_elementwise_range_special() -> None:
    e = nwp.int_range(0, 100)
    assert isinstance(e._ir, ir.RangeExpr)
    e = nwp.int_range(nwp.len(), dtype=nw.UInt32).alias("index")
    e_ir = e._ir
    assert isinstance(e_ir, ir.Alias)
    assert isinstance(e_ir.expr, ir.RangeExpr)
    assert isinstance(e_ir.expr.input[0], ir.Lit)
    assert isinstance(e_ir.expr.input[1], ir.Len)


def test_function_arity_invalid() -> None:
    expr_s = re.escape("col('a').first()")
    function = re.escape("int_range()")
    pattern = re_compile(rf"Expected 2 inputs for `{function}`, got 1:\n  {expr_s}")
    with pytest.raises(TypeError, match=pattern):
        IntRange(step=1, dtype=nw.Int64()).to_function_expr(agg.First(expr=ir.col("a")))


def test_int_range_invalid() -> None:
    with pytest.raises(ShapeError, match=r"int_range.+non-scalar.+col"):
        nwp.int_range(nwp.col("a"))
    with pytest.raises(ShapeError, match=r"int_range.+non-scalar.+by_index"):
        nwp.int_range(nwp.nth(1), 10)
    with pytest.raises(ShapeError, match=r"int_range.+non-scalar.+filter"):
        nwp.int_range(nwp.col("a").filter(nwp.col("a") == 1), 100)
    with pytest.raises(ShapeError, match=r"int_range.+non-scalar.+abs"):
        nwp.int_range(0, nwp.col("a").abs())
    with pytest.raises(ShapeError, match=r"int_range.+non-scalar.+\+"):
        nwp.int_range(nwp.col("a") + 1)
    with pytest.raises(ShapeError, match=r"int_range.+non-scalar.+keep"):
        nwp.int_range((1 + nwp.col("b")).name.keep())


def test_date_range_invalid() -> None:
    start, end = dt.date(2000, 1, 1), dt.date(2001, 1, 1)
    with pytest.raises(ShapeError, match=r"date_range.+non-scalar.+col"):
        nwp.date_range(nwp.col("a"), nwp.col("b"))
    with pytest.raises(ShapeError, match=r"date_range.+non-scalar.+all"):
        nwp.date_range(start, nwp.all())
    with pytest.raises(TypeError, match=r"`closed` must be one of.+, got.+middle"):
        nwp.date_range(start, end, closed="middle")  # type: ignore[call-overload]
    with pytest.raises(
        ComputeError, match="`interval` input for `date_range` must consist of full days"
    ):
        nwp.date_range(start, end, interval="24h")
    with pytest.raises(NotImplementedError, match=r"not support.+'mo'.+yet"):
        nwp.date_range(start, end, interval="1mo")
    with pytest.raises(NotImplementedError, match=r"not support.+'q'.+yet"):
        nwp.date_range(start, end, interval="2q")
    with pytest.raises(NotImplementedError, match=r"not support.+'y'.+yet"):
        nwp.date_range(start, end, interval="3y")


def test_int_range_eager_invalid() -> None:
    with pytest.raises(InvalidOperationError):
        nwp.int_range(nwp.len(), eager="pyarrow")  # type: ignore[call-overload]
    with pytest.raises(InvalidOperationError):
        nwp.int_range(10, nwp.col("a").last(), eager=Implementation.PYARROW)  # type: ignore[call-overload]
    with pytest.raises(NotImplementedError):
        nwp.int_range(10, eager="pandas")
    with pytest.raises(NotImplementedError, match="duckdb"):
        nwp.int_range(10, eager="duckdb")  # type: ignore[call-overload]


def test_date_range_eager_invalid() -> None:
    start, end = dt.date(2000, 1, 1), dt.date(2001, 1, 1)

    with pytest.raises(InvalidOperationError):
        nwp.date_range(1, end, eager="pyarrow")  # type: ignore[call-overload]
    with pytest.raises(InvalidOperationError):
        nwp.date_range(start, nwp.col("a").last(), eager=Implementation.PYARROW)  # type: ignore[call-overload]
    with pytest.raises(NotImplementedError):
        nwp.date_range(start, end, eager="cudf")
    with pytest.raises(NotImplementedError, match="sqlframe"):
        nwp.date_range(start, end, eager="sqlframe")  # type: ignore[call-overload]


def test_over_invalid() -> None:
    with pytest.raises(TypeError, match=r"one of.+partition_by.+or.+order_by"):
        nwp.col("a").last().over()

    # NOTE: Non-`polars` rule
    pattern = re.compile(r"cannot use.+over.+elementwise", re.IGNORECASE)
    with pytest.raises(InvalidOperationError, match=pattern):
        nwp.col("a").fill_null(3).over("b")

    # NOTE: This version isn't elementwise
    expr_ir = nwp.col("a").fill_null(strategy="backward").over("b")._ir
    assert isinstance(expr_ir, ir.Over)
    assert isinstance(expr_ir.expr, ir.FunctionExpr)
    assert isinstance(expr_ir.expr.function, F.FillNullWithStrategy)


def test_over_nested() -> None:
    pattern = re.compile(r"cannot nest.+over", re.IGNORECASE)
    with pytest.raises(InvalidOperationError, match=pattern):
        nwp.col("a").mean().over("b").over("c")
    with pytest.raises(InvalidOperationError, match=pattern):
        nwp.col("a").mean().over("b").over("c", order_by="i")


# NOTE: This *can* error in polars, but only if the length **actually changes**
# The rule then breaks down to needing the same length arrays in all parts of the over
def test_over_filtration() -> None:
    pattern = re.compile(r"cannot use.+over.+change length", re.IGNORECASE)
    with pytest.raises(InvalidOperationError, match=pattern):
        nwp.col("a").drop_nulls().over("b")
    with pytest.raises(InvalidOperationError, match=pattern):
        nwp.col("a").drop_nulls().over("b", order_by="i")
    with pytest.raises(InvalidOperationError, match=pattern):
        nwp.col("a").diff().drop_nulls().over("b", order_by="i")


@pytest.mark.parametrize(
    "left",
    [nwp.col("a"), nwp.len(), nwp.lit(1), nwp.max("a"), nwp.sum_horizontal("a", "b")],
    ids=["col", "len", "lit", "agg_expr", "horizontal"],
)
@pytest.mark.parametrize(
    "right",
    [
        nwp.col("c").null_count(),
        nwp.col("c").fill_null(1),
        nwp.col("c").rolling_mean(5),
        nwp.nth(1).cum_min(),
        nwp.col("c").shift(2),
    ],
    ids=[
        "aggregation",
        "elementwise",
        "length_preserving_1",
        "length_preserving_2",
        "length_preserving_3",
    ],
)
def test_binary_expr(left: nwp.Expr, right: nwp.Expr) -> None:
    assert _is_expr_ir_binary_expr(left // right)
    assert _is_expr_ir_binary_expr(right * left)


a = nwp.col("a")
b = nwp.col("b").exp()
filtered = a.filter(a=1)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (a.unique(), b.unique()),
        (a.mode(), b.unique()),
        (a.drop_nulls(), b.mode()),
        (a.gather_every(2, 1), b.drop_nulls()),
        (a.map_batches(lambda x: x), b.gather_every(1, 0)),
        (filtered, b.unique()),
    ],
)
def test_binary_expr_multi_length_changing(left: nwp.Expr, right: nwp.Expr) -> None:
    pattern = re_compile(r"length-changing.+used in isolation.+or.+aggregation")
    ctx = pytest.raises(InvalidOperationError, match=pattern)
    with ctx:
        left + right
    with ctx:
        right / left


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (a.unique(), b),
        (a.map_batches(lambda x: x, is_elementwise=True), b.gather_every(1, 0)),
        (a, b.mode()),
        (a.gather_every(2, 1), nwp.nth(-1)),
        (a.hist(bin_count=4), b.rolling_mean(5)),
        (filtered, b.cum_prod()),
        (filtered.alias("aaaaa").sort(), b),
        (b.cum_max(), filtered.cast(nw.Int64)),
    ],
)
def test_binary_expr_mixed_length_changing(left: nwp.Expr, right: nwp.Expr) -> None:
    pattern = re_compile(r"Cannot.+length-changing.+length-preserving")
    ctx = pytest.raises(InvalidOperationError, match=pattern)
    with ctx:
        left - right
    with ctx:
        right * left


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
    assert _is_expr_ir_binary_expr(
        a.filter(a.is_last_distinct()).first() ^ b.filter(b.is_not_null())
    )
    assert _is_expr_ir_binary_expr(
        a.filter(a=a.last()) <= b.filter(b.is_not_nan()).median()
    )


def test_map_batches_invalid() -> None:
    with pytest.raises(
        TypeError,
        match=r"A function cannot both return a scalar and preserve length, they are mutually exclusive",
    ):
        nwp.col("a").map_batches(lambda x: x, is_elementwise=True, returns_scalar=True)


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
    assert e_ir.function.other.value.to_native() is native


@pytest.mark.parametrize(
    ("other", "context"),
    [
        ("words", pytest.raises(TypeError, match=r"str \| bytes.+str")),
        (b"words", pytest.raises(TypeError, match=r"str \| bytes.+bytes")),
        (
            999,
            pytest.raises(
                TypeError, match=re.compile(r"only.+iterable.+int", re.IGNORECASE)
            ),
        ),
    ],
)
def test_is_in_invalid(other: Any, context: AbstractContextManager[Any]) -> None:
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
        ((), {}, pytest.raises(TypeError, match=re_compile(r"at least one predicate"))),
        ((nwp.col("b") > 1, nwp.col("c").is_null()), {}, nullcontext()),
        (
            ([nwp.col("b") > 1], nwp.col("c").is_null()),
            {},
            pytest.raises(
                TypeError, match=re_compile(r"Expr.+ is not supported in `nw.lit`")
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
    assert isinstance(e_ir, ir.LitSeries)
    assert isinstance(e_ir.dtype, nw.String)
    assert isinstance(e_ir.value, nwp.Series)
    assert isinstance(e_ir.value.to_native(), pa.ChunkedArray)
    assert e_ir.value.to_list() == data


def test_lit_invalid() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    pattern = re_compile(r"is not supported in `nw.lit`")
    context = pytest.raises(TypeError, match=pattern)
    with context:
        nwp.lit(object())  # type: ignore[arg-type]
    with context:
        nwp.lit({1, 2, 3})  # type: ignore[arg-type]

    # NOTE: The overloads are broken
    # this gets the right error without making the LSP blow up
    value: Any = 1
    bad: pa.Scalar[Any] = pa.scalar(value)
    with context:
        nwp.lit(bad)  # type: ignore[arg-type]
    with context:
        nwp.lit([bad])


@pytest.mark.parametrize(
    ("value", "dtype", "expected"),
    [
        pytest.param((), nw.List(nw.Int32()), nw.List(nw.Int32()), id="list-empty-tuple"),
        pytest.param(
            [], nw.List(nw.Binary()), nw.List(nw.Binary()), id="list-empty-list"
        ),
        pytest.param({}, nw.Struct({}), nw.Struct([]), id="struct-empty-dict"),
        pytest.param(("foo", "bar"), None, nw.List(nw.String()), id="list-str"),
        pytest.param([1.0, 2.2], None, nw.List(nw.Float64()), id="list-float"),
        pytest.param(
            {"field_1": 42}, None, nw.Struct({"field_1": nw.Int64()}), id="struct-int"
        ),
        pytest.param(
            (None, dt.time(1, 1, 1), dt.time(2, 1, 1)),
            None,
            nw.List(nw.Time()),
            id="list-time",
        ),
        pytest.param(
            [dt.date(2000, 1, 1), None, None], None, nw.List(nw.Date()), id="list-date"
        ),
        pytest.param(
            {"field_1": dt.datetime(2002, 1, 1)},
            None,
            nw.Struct({"field_1": nw.Datetime()}),
            id="struct-datetime",
        ),
        pytest.param(
            {"field_1": 42, "field_2": 1.2, "field_3": True},
            nw.Struct(
                {"field_1": nw.Int32(), "field_2": nw.Float64(), "field_3": nw.Boolean()}
            ),
            nw.Struct(
                {"field_1": nw.Int32(), "field_2": nw.Float64(), "field_3": nw.Boolean()}
            ),
            id="struct-multiple",
        ),
    ],
)
def test_lit_nested(
    value: PythonLiteral, dtype: IntoDType | None, expected: DType
) -> None:
    """Adapted from [`test_nested_structures`].

    Just covering the inference + error handling.

    [`test_nested_structures`]: https://github.com/narwhals-dev/narwhals/blob/228fee8d83d92d06e6cb32646d0e131acf0c1e2e/tests/expr_and_series/lit_test.py#L142-L209
    """
    lit = nwp.lit(value, dtype)
    assert lit.meta.is_literal()
    e_ir = lit._ir
    assert isinstance(e_ir, ir.Lit)
    assert e_ir.dtype == expected
    assert hash(e_ir) == hash(nwp.lit(value, dtype)._ir)


@pytest.mark.parametrize("value", [[], (), {}], ids=["list", "tuple", "dict"])
def test_lit_nested_empty_invalid(value: PythonLiteral) -> None:
    msg = "Cannot infer dtype for empty nested structure. Please provide an explicit dtype parameter."
    with pytest.raises(TypeError, match=msg):
        nwp.lit(value)
    with pytest.raises(TypeError, match=msg):
        nwp.col("a").sort_by(value, nwp.lit(1))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value", [[None, None], (None,), {"a": 1, "b": None}], ids=["list", "tuple", "dict"]
)
def test_lit_nested_inner_invalid(value: PythonLiteral) -> None:
    msg = "Nested dtypes containing nulls are not yet supported"
    with pytest.raises(TypeError, match=msg):
        nwp.lit(value)


# TODO @dangotbanned: Mix up the dtypes
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # List containing nested structures
        ([[1, 2], [3, 4]], List(List(Int64()))),
        ([(1, 2), (3, 4)], List(List(Int64()))),
        ([{"a": 1}, {"a": 2}], List(Struct({"a": Int64()}))),
        # Tuple containing nested structures
        (([1, 2], [3, 4]), List(List(Int64()))),
        (((1, 2), (3, 4)), List(List(Int64()))),
        (({"a": 1}, {"a": 2}), List(Struct({"a": Int64()}))),
        # Dict containing nested structures
        ({"a": [1, 2], "b": [3, 4]}, Struct({"a": List(Int64()), "b": List(Int64())})),
        ({"a": (1, 2), "b": (3, 4)}, Struct({"a": List(Int64()), "b": List(Int64())})),
        (
            {"a": {"x": 1}, "b": {"y": 2}},
            Struct({"a": Struct({"x": Int64()}), "b": Struct({"y": Int64()})}),
        ),
    ],
    ids=str,
)
def test_lit_nested_inception(
    value: dict[str, Any] | list[Any] | tuple[Any, ...], expected: DType
) -> None:
    lit = nwp.lit(value)
    assert lit.meta.is_literal()
    e_ir = lit._ir
    assert isinstance(e_ir, ir.Lit)
    assert e_ir.dtype == expected
    assert hash(e_ir) == hash(nwp.lit(value)._ir)


def test_sort_by_empty() -> None:
    with pytest.raises(TypeError, match=re_compile("at least one sort key")):
        nwp.col("a").sort_by(())


a, b = nwp.col("a"), nwp.col("b")

if find_spec("polars"):
    _s = nwp.Series.from_iterable([1, 2, 3], backend="polars")
    case_lit_series: Seq[Any] = ((nwp.lit(_s), ()),)
    id_lit_series: Seq[str] = ("lit-series",)
else:  # pragma: no cover
    case_lit_series, id_lit_series = (), ()


@pytest.mark.parametrize(
    ("by", "more_by"),
    [
        (1, ()),
        (nwp.len(), ()),
        (b.first().sort_by("d"), ()),
        (nwp.lit(1).alias("bad"), []),
        ([1, 2, 3], ("a", b)),
        ([(nwp.len() * 1).name.keep()], ()),
        ([b, nwp.when(a).then(b.alias("c")), ncs.last().kurtosis()], ()),
        (ncs.last(), ("c", "d", b.mode(keep="any"))),
        (nwp.int_range(2).sort(), ()),
        (b.filter(c=2), ()),
        (b.drop_nulls().name.prefix("before_"), ()),
        (a + b, (nwp.when(b.min().cast(nw.Boolean)).then(1),)),
        *case_lit_series,
    ],
    ids=[
        "scalar",
        "len",
        "agg",
        "lit",
        "lit-list",
        "iterable-len",
        "selector-agg",
        "agg-function",
        "range",
        "filter",
        "row-separable",
        "when-scalar",
        *id_lit_series,
    ],
)
def test_sort_by_invalid(
    by: OneOrIterable[nwp.Expr | nwp.Selector | str],
    more_by: Iterable[nwp.Expr | nwp.Selector | str],
) -> None:
    pattern = re_compile(r"all.+sort_by.+must be length-preserving")
    a = nwp.col("a")
    with pytest.raises(InvalidOperationError, match=pattern):
        a.sort_by(by, *more_by)


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
    include_breakpoint_default = False
    a = nwp.col("a")
    hist_1 = a.hist(
        bin_count=bin_count_default, include_breakpoint=include_breakpoint_default
    )
    hist_2 = a.hist()
    hist_3 = a.hist(bin_count=5)
    hist_4 = a.hist(include_breakpoint=True)

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
        nwp.min("a").over(pl.col("b"))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "base",
    [
        nwp.when,
        nwp.when(a="b").then(1).when,
        nwp.when(nwp.col("a").is_finite())
        .then(1)
        .when((nwp.median("a", "b", "c") > 2) | nwp.col("d").is_nan())
        .then(2)
        .when,
    ],
    ids=["when", "Then", "ChainedThen"],
)
def test_when_empty(base: Callable[..., When | ChainedWhen]) -> None:
    at_least_one = pytest.raises(TypeError, match=re_compile(r"at least one predicate"))
    with at_least_one:
        base()
    with at_least_one:
        base([])
    with at_least_one:
        base(iter(()))


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


def test_dt_timestamp_invalid() -> None:
    assert nwp.col("a").dt.timestamp()
    with pytest.raises(
        TypeError, match=re_compile(r"only.+time unit.+supported.+Got: 's'")
    ):
        nwp.col("a").dt.timestamp("s")


def test_dt_truncate_invalid() -> None:
    assert nwp.col("a").dt.truncate("1d")
    with pytest.raises(ValueError, match=re_compile(r"invalid.+every.+abcd")):
        nwp.col("a").dt.truncate("abcd")


def test_replace_strict() -> None:
    a = nwp.col("a")
    remapping = a.replace_strict({1: 3, 2: 4}, return_dtype=nw.Int8)
    sequences = a.replace_strict(old=[1, 2], new=[3, 4], return_dtype=nw.Int8())
    assert_expr_ir_equal(remapping, sequences)


def test_replace_strict_invalid() -> None:
    with pytest.raises(
        TypeError,
        match="`new` argument is required if `old` argument is not a Mapping type",
    ):
        nwp.col("a").replace_strict("b")

    with pytest.raises(
        TypeError,
        match="`new` argument cannot be used if `old` argument is a Mapping type",
    ):
        nwp.col("a").replace_strict(old={1: 2, 3: 4}, new=[5, 6, 7])


def test_mode_invalid() -> None:
    with pytest.raises(
        TypeError, match=r"keep.+must be one of.+all.+any.+but got 'first'"
    ):
        nwp.col("a").mode(keep="first")  # type: ignore[arg-type]


def test_broadcast_len_1_series_invalid() -> None:
    pytest.importorskip("pyarrow")
    data = {"a": [1, 2, 3]}
    values = [4]
    df = nwp.DataFrame.from_dict(data, backend="pyarrow")
    ser = nwp.Series.from_iterable(values, name="bad", backend="pyarrow")
    with pytest.raises(
        ShapeError,
        match=re_compile(
            r"series.+bad.+length.+1.+match.+DataFrame.+height.+3.+broadcasted.+\.first\(\)"
        ),
    ):
        df.with_columns(ser)

    expected_series = {"a": [1, 2, 3], "literal": [4, 4, 4]}
    # we can only preserve `Series.name` if we got a `lit(Series).first()`, not `lit(Series.first())`
    expected_series_literal = {"a": [1, 2, 3], "bad": [4, 4, 4]}

    assert_equal_data(df.with_columns(ser.first()), expected_series)
    assert_equal_data(df.with_columns(ser.last()), expected_series)
    assert_equal_data(df.with_columns(nwp.lit(ser).first()), expected_series_literal)


@pytest.mark.parametrize(
    ("window_size", "min_samples", "context"),
    [
        (-1, None, pytest.raises(ValueError, match=r"window_size.+>= 1")),
        (2, -1, pytest.raises(ValueError, match=r"min_samples.+>= 1")),
        (
            1,
            2,
            pytest.raises(InvalidOperationError, match=r"min_samples.+<=.+window_size"),
        ),
        (
            4.2,
            None,
            pytest.raises(TypeError, match=r"Expected.+int.+got.+float.+\s+window_size="),
        ),
        (
            2,
            4.2,
            pytest.raises(TypeError, match=r"Expected.+int.+got.+float.+\s+min_samples="),
        ),
    ],
)
def test_rolling_expr_invalid(
    window_size: int, min_samples: int | None, context: pytest.RaisesExc[Any]
) -> None:
    a = nwp.col("a")
    with context:
        a.rolling_sum(window_size, min_samples=min_samples)
    with context:
        a.rolling_mean(window_size, min_samples=min_samples)
    with context:
        a.rolling_var(window_size, min_samples=min_samples)
    with context:
        a.rolling_std(window_size, min_samples=min_samples)


def test_list_contains_invalid() -> None:
    a = nwp.col("a")

    ok = a.list.contains("a")
    assert_expr_ir_equal(
        ok,
        ir.FunctionExpr(
            input=(ir.col("a"), ir.lit("a", nw.String)), function=ir.lists.Contains()
        ),
    )
    assert a.list.contains(a.first())
    assert a.list.contains(1)
    assert a.list.contains(nwp.lit(1))
    assert a.list.contains(dt.datetime(2000, 2, 1, 9, 26, 5))
    assert a.list.contains(a.abs().fill_null(5).mode(keep="any"))

    with pytest.raises(ShapeError, match=r"list.contains.+non-scalar.+`col\('a'\)"):
        a.list.contains(a)

    with pytest.raises(ShapeError, match=r"list.contains.+non-scalar.+abs"):
        a.list.contains(a.abs())


def test_list_get_invalid() -> None:
    a = nwp.col("a")
    assert a.list.get(0)
    pattern = re_compile(r"expected.+int.+got.+str.+'not an index'")
    with pytest.raises(TypeError, match=pattern):
        a.list.get("not an index")  # type: ignore[arg-type]
    pattern = re_compile(r"index.+out of bounds.+>= 0.+got -1")
    with pytest.raises(InvalidOperationError, match=pattern):
        a.list.get(-1)
