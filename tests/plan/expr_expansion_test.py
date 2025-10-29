from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable

import pytest

import narwhals as nw
from narwhals import _plan as nwp
from narwhals._plan import expressions as ir, selectors as ncs
from narwhals._utils import zip_strict
from narwhals.exceptions import (
    ColumnNotFoundError,
    DuplicateError,
    MultiOutputExpressionError,
)
from tests.plan.utils import Frame, assert_expr_ir_equal, named_ir, re_compile

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from narwhals._plan.typing import IntoExpr, MapIR
    from narwhals.dtypes import DType
    from narwhals.typing import IntoSchema


@pytest.fixture(scope="module")
def schema_1() -> dict[str, DType]:
    return {
        "a": nw.Int64(),
        "b": nw.Int32(),
        "c": nw.Int16(),
        "d": nw.Int8(),
        "e": nw.UInt64(),
        "f": nw.UInt32(),
        "g": nw.UInt16(),
        "h": nw.UInt8(),
        "i": nw.Float64(),
        "j": nw.Float32(),
        "k": nw.String(),
        "l": nw.Datetime(),
        "m": nw.Boolean(),
        "n": nw.Date(),
        "o": nw.Datetime(),
        "p": nw.Categorical(),
        "q": nw.Duration(),
        "r": nw.Enum(["A", "B", "C"]),
        "s": nw.List(nw.String()),
        "u": nw.Struct({"a": nw.Int64(), "k": nw.String()}),
    }


@pytest.fixture(scope="module")
def df_1(schema_1: IntoSchema) -> Frame:
    return Frame.from_mapping(schema_1)


MULTI_OUTPUT_EXPRS = (
    pytest.param(nwp.col("a", "b", "c")),
    pytest.param(ncs.numeric() - ncs.matches("[d-j]")),
    pytest.param(nwp.nth(0, 1, 2)),
    pytest.param(ncs.by_dtype(nw.Int64, nw.Int32, nw.Int16)),
    pytest.param(ncs.by_name("a", "b", "c")),
)
"""All of these resolve to `["a", "b", "c"]`."""

BIG_EXCLUDE = nwp.exclude("k", "l", "m", "n", "o", "p", "s", "u", "r", "a", "b", "e", "q")


def udf_name_map(name: str) -> str:
    original = name
    upper = name.upper()
    lower = name.lower()
    title = name.title()
    return f"{original=} | {upper=} | {lower=} | {title=}"


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nwp.col("a").name.to_uppercase(), "A"),
        (nwp.col("B").name.to_lowercase(), "b"),
        (nwp.col("c").name.suffix("_after"), "c_after"),
        (nwp.col("d").name.prefix("before_"), "before_d"),
        (
            nwp.col("aBcD EFg hi").name.map(udf_name_map),
            "original='aBcD EFg hi' | upper='ABCD EFG HI' | lower='abcd efg hi' | title='Abcd Efg Hi'",
        ),
        (nwp.col("a").min().alias("b").over("c").alias("d").max().name.keep(), "a"),
        (
            (
                nwp.col("hello")
                .sort_by(nwp.col("ignore me"))
                .max()
                .over("ignore me as well")
                .first()
                .name.to_uppercase()
            ),
            "HELLO",
        ),
        pytest.param(
            (
                nwp.col("start")
                .alias("next")
                .sort()
                .round()
                .fill_null(5)
                .alias("noise")
                .name.suffix("_end")
            ),
            "noise_end",
        ),
    ],
)
def test_special_aliases_single(expr: nwp.Expr, expected: str) -> None:
    df = Frame.from_names(
        "a",
        "B",
        "c",
        "d",
        "aBcD EFg hi",
        "hello",
        "start",
        "ignore me",
        "ignore me as well",
        "unreferenced",
    )
    df.assert_selects(expr, expected)


def alias_replace_guarded(name: str) -> MapIR:
    """Guards against repeatedly creating the same alias."""

    def fn(e_ir: ir.ExprIR) -> ir.ExprIR:
        if isinstance(e_ir, ir.Alias) and e_ir.name != name:
            return ir.Alias(expr=e_ir.expr, name=name)
        return e_ir

    return fn


def alias_replace_unguarded(name: str) -> MapIR:
    """**Does not guard against recursion**!

    Handling the recursion stopping **should be** part of the impl of `ExprIR.map_ir`.

    - *Ideally*, return an identical result to `alias_replace_guarded` (after the same number of iterations)
    - *Pragmatically*, it might require an extra iteration to detect a cycle
    """

    def fn(e_ir: ir.ExprIR) -> ir.ExprIR:
        if isinstance(e_ir, ir.Alias):
            return ir.Alias(expr=e_ir.expr, name=name)
        return e_ir

    return fn


@pytest.mark.parametrize(
    ("expr", "function", "expected"),
    [
        (nwp.col("a"), alias_replace_guarded("never"), nwp.col("a")),
        (nwp.col("a"), alias_replace_unguarded("never"), nwp.col("a")),
        (nwp.col("a").alias("b"), alias_replace_guarded("c"), nwp.col("a").alias("c")),
        (nwp.col("a").alias("b"), alias_replace_unguarded("c"), nwp.col("a").alias("c")),
        (
            nwp.col("a").alias("d").first().over("b", order_by="c").alias("e"),
            alias_replace_guarded("d"),
            nwp.col("a").alias("d").first().over("b", order_by="c").alias("d"),
        ),
        (
            nwp.col("a").alias("d").first().over("b", order_by="c").alias("e"),
            alias_replace_unguarded("d"),
            nwp.col("a").alias("d").first().over("b", order_by="c").alias("d"),
        ),
        (
            nwp.col("a").alias("e").abs().alias("f").sort().alias("g"),
            alias_replace_guarded("e"),
            nwp.col("a").alias("e").abs().alias("e").sort().alias("e"),
        ),
        (
            nwp.col("a").alias("e").abs().alias("f").sort().alias("g"),
            alias_replace_unguarded("e"),
            nwp.col("a").alias("e").abs().alias("e").sort().alias("e"),
        ),
    ],
)
def test_map_ir_recursive(expr: nwp.Expr, function: MapIR, expected: nwp.Expr) -> None:
    actual = expr._ir.map_ir(function)
    assert_expr_ir_equal(actual, expected)


def test_expand_selectors_funky_1(df_1: Frame) -> None:
    # root->selection->transform
    selector = ncs.matches("[a-m]") & ~ncs.numeric()
    expr = selector.sort(nulls_last=True).first() != nwp.lit(None)
    expecteds = [
        named_ir(name, nwp.col(name).sort(nulls_last=True).first() != nwp.lit(None))
        for name in ("k", "l", "m")
    ]
    actuals = df_1.project(expr)
    for actual, expected in zip_strict(actuals, expecteds):
        assert_expr_ir_equal(actual, expected)


def test_expand_selectors_funky_2(df_1: Frame) -> None:
    # root->selection->transform
    # leaf->selection
    expr = ncs.numeric().mean().over("k", order_by=ncs.by_dtype(nw.Date) | ncs.boolean())
    root_names = "a", "b", "c", "d", "e", "f", "g", "h", "i", "j"
    expecteds = (
        named_ir(name, nwp.col(name).mean().over("k", order_by=("m", "n")))
        for name in root_names
    )
    actuals = df_1.project(expr)
    for actual, expected in zip_strict(actuals, expecteds):
        assert_expr_ir_equal(actual, expected)


def test_expand_selectors_funky_3(df_1: Frame) -> None:
    # root->selection->transform->rename
    # leaf->selection
    expr = (
        ncs.datetime()
        .dt.timestamp()
        .min()
        .over(ncs.string() | ncs.boolean())
        .last()
        .name.to_uppercase()
    )
    expecteds = [
        named_ir(name.upper(), nwp.col(name).dt.timestamp().min().over("k", "m").last())
        for name in ("l", "o")
    ]
    actuals = df_1.project(expr)
    for actual, expected in zip_strict(actuals, expecteds):
        assert_expr_ir_equal(actual, expected)


@pytest.mark.parametrize(
    ("into_exprs", "expected"),
    [
        pytest.param("a", [nwp.col("a")], id="Col"),
        pytest.param(
            nwp.col("b", "c", "d"),
            [nwp.col("b"), nwp.col("c"), nwp.col("d")],
            id="ByName(3)",
        ),
        pytest.param(nwp.nth(6), [nwp.col("g")], id="ByIndex(1)"),
        pytest.param(
            nwp.nth(9, 8, -5), [nwp.col("j"), nwp.col("i"), nwp.col("p")], id="ByIndex(3)"
        ),
        pytest.param(
            [nwp.nth(2).alias("c again"), nwp.nth(-1, -2).name.to_uppercase()],
            [
                named_ir("c again", nwp.col("c")),
                named_ir("U", nwp.col("u")),
                named_ir("S", nwp.col("s")),
            ],
            id="ByIndex(1)-Alias-ByIndex(2)-Uppercase",
        ),
        pytest.param(
            nwp.all(),
            [
                nwp.col("a"),
                nwp.col("b"),
                nwp.col("c"),
                nwp.col("d"),
                nwp.col("e"),
                nwp.col("f"),
                nwp.col("g"),
                nwp.col("h"),
                nwp.col("i"),
                nwp.col("j"),
                nwp.col("k"),
                nwp.col("l"),
                nwp.col("m"),
                nwp.col("n"),
                nwp.col("o"),
                nwp.col("p"),
                nwp.col("q"),
                nwp.col("r"),
                nwp.col("s"),
                nwp.col("u"),
            ],
            id="All",
        ),
        pytest.param(
            (ncs.numeric() - ncs.by_dtype(nw.Float32(), nw.Float64()))
            .cast(nw.Int64)
            .mean()
            .name.suffix("_mean"),
            [
                named_ir("a_mean", nwp.col("a").cast(nw.Int64()).mean()),
                named_ir("b_mean", nwp.col("b").cast(nw.Int64()).mean()),
                named_ir("c_mean", nwp.col("c").cast(nw.Int64()).mean()),
                named_ir("d_mean", nwp.col("d").cast(nw.Int64()).mean()),
                named_ir("e_mean", nwp.col("e").cast(nw.Int64()).mean()),
                named_ir("f_mean", nwp.col("f").cast(nw.Int64()).mean()),
                named_ir("g_mean", nwp.col("g").cast(nw.Int64()).mean()),
                named_ir("h_mean", nwp.col("h").cast(nw.Int64()).mean()),
            ],
            id="Selector-SUB-Cast-Mean-Suffix",
        ),
        pytest.param(
            nwp.col("u").alias("1").alias("2").alias("3").alias("4").name.keep(),
            [named_ir("u", nwp.col("u"))],
            id="Alias-Etc-Keep",
        ),
        pytest.param(
            (
                (ncs.numeric() ^ (ncs.matches(r"[abcdg]") | ncs.by_name("i", "f"))) * 100
            ).name.suffix("_mult_100"),
            [
                named_ir("e_mult_100", (nwp.col("e") * nwp.lit(100))),
                named_ir("h_mult_100", (nwp.col("h") * nwp.lit(100))),
                named_ir("j_mult_100", (nwp.col("j") * nwp.lit(100))),
            ],
            id="Selector-XOR-OR-BinaryExpr-Suffix",
        ),
        pytest.param(
            ncs.by_dtype(nw.Duration())
            .dt.total_minutes()
            .name.map(lambda nm: f"total_mins: {nm!r} ?"),
            [named_ir("total_mins: 'q' ?", nwp.col("q").dt.total_minutes())],
            id="ByDType-TotalMins-Name-Map",
        ),
        pytest.param(
            nwp.col("f", "g")
            .cast(nw.String)
            .str.starts_with("1")
            .all()
            .name.suffix("_all_starts_with_1"),
            [
                named_ir(
                    "f_all_starts_with_1",
                    nwp.col("f").cast(nw.String).str.starts_with("1").all(),
                ),
                named_ir(
                    "g_all_starts_with_1",
                    nwp.col("g").cast(nw.String).str.starts_with("1").all(),
                ),
            ],
            id="ByName(2)-Cast-StartsWith-All-Suffix",
        ),
        pytest.param(
            nwp.col("a", "b")
            .first()
            .over("c", "e", order_by="d")
            .name.suffix("_first_over_part_order_1"),
            [
                named_ir(
                    "a_first_over_part_order_1",
                    nwp.col("a")
                    .first()
                    .over(nwp.col("c"), nwp.col("e"), order_by=[nwp.col("d")]),
                ),
                named_ir(
                    "b_first_over_part_order_1",
                    nwp.col("b")
                    .first()
                    .over(nwp.col("c"), nwp.col("e"), order_by=[nwp.col("d")]),
                ),
            ],
            id="ByName(2)-First-Over-Partitioned-Ordered-Suffix",
        ),
        pytest.param(
            BIG_EXCLUDE,
            [
                nwp.col("c"),
                nwp.col("d"),
                nwp.col("f"),
                nwp.col("g"),
                nwp.col("h"),
                nwp.col("i"),
                nwp.col("j"),
            ],
            id="Exclude",
        ),
        pytest.param(
            BIG_EXCLUDE.name.suffix("_2"),
            [
                named_ir("c_2", nwp.col("c")),
                named_ir("d_2", nwp.col("d")),
                named_ir("f_2", nwp.col("f")),
                named_ir("g_2", nwp.col("g")),
                named_ir("h_2", nwp.col("h")),
                named_ir("i_2", nwp.col("i")),
                named_ir("j_2", nwp.col("j")),
            ],
            id="Exclude-Suffix",
        ),
        pytest.param(
            nwp.col("c").alias("c_min_over_order_by").min().over(order_by=ncs.string()),
            [
                named_ir(
                    "c_min_over_order_by",
                    nwp.col("c").min().over(order_by=[nwp.col("k")]),
                )
            ],
            id="Alias-Min-Over-Order-By-Selector",
        ),
        pytest.param(
            (ncs.by_name("a", "b", "c") / nwp.col("e").first())
            .over("g", "f", order_by="f")
            .name.prefix("hi_"),
            [
                named_ir(
                    "hi_a",
                    (nwp.col("a") / nwp.col("e").first()).over("g", "f", order_by="f"),
                ),
                named_ir(
                    "hi_b",
                    (nwp.col("b") / nwp.col("e").first()).over("g", "f", order_by="f"),
                ),
                named_ir(
                    "hi_c",
                    (nwp.col("c") / nwp.col("e").first()).over("g", "f", order_by="f"),
                ),
            ],
            id="Selector-BinaryExpr-Over-Prefix",
        ),
        pytest.param(
            [
                nwp.col("c").sort_by(nwp.col("c", "i")).first().alias("ByName(2)"),
                nwp.col("c").sort_by("c", "i").first().alias("Column_x2"),
            ],
            [
                named_ir(
                    "ByName(2)", nwp.col("c").sort_by(nwp.col("c"), nwp.col("i")).first()
                ),
                named_ir(
                    "Column_x2", nwp.col("c").sort_by(nwp.col("c"), nwp.col("i")).first()
                ),
            ],
            id="SortBy-ByName",
        ),
        pytest.param(
            nwp.nth(1).mean().over("k", order_by=nwp.nth(4, 5)),
            [
                nwp.col("b")
                .mean()
                .over(nwp.col("k"), order_by=(nwp.col("e"), nwp.col("f")))
            ],
            id="Over-OrderBy-ByIndex(2)",
        ),
        pytest.param(
            nwp.col("f").max().over(ncs.by_dtype(nw.Date, nw.Datetime)),
            [nwp.col("f").max().over(nwp.col("l"), nwp.col("n"), nwp.col("o"))],
            id="Over-Partitioned-Selector",
        ),
    ],
)
def test_prepare_projection(
    into_exprs: IntoExpr | Sequence[IntoExpr], expected: Sequence[nwp.Expr], df_1: Frame
) -> None:
    actual = df_1.project(into_exprs)
    assert len(actual) == len(expected)
    for lhs, rhs in zip(actual, expected):
        assert_expr_ir_equal(lhs, rhs)


@pytest.mark.parametrize(
    "expr",
    [
        nwp.all(),
        nwp.nth(1, 2, 3),
        nwp.col("a", "b", "c"),
        ncs.boolean() | ncs.categorical(),
        (ncs.by_name("a", "b") | ncs.string()),
        (nwp.col("b", "c") & nwp.col("a")),
        nwp.col("a", "b").min().over("c", order_by="e"),
        (~ncs.by_dtype(nw.Int64()) - ncs.datetime()),
        nwp.nth(6, 2).abs().cast(nw.Int32()) + 10,
        *MULTI_OUTPUT_EXPRS,
    ],
)
def test_prepare_projection_duplicate(expr: nwp.Expr, df_1: Frame) -> None:
    pattern = re.compile(r"\.alias\(.dupe.\)")
    expr = expr.alias("dupe")
    with pytest.raises(DuplicateError, match=pattern):
        df_1.project(expr)


@pytest.mark.parametrize(
    ("into_exprs", "missing"),
    [
        ([nwp.col("y", "z")], ["y", "z"]),
        ([nwp.col("a", "b", "z")], ["z"]),
        ([nwp.col("x", "b", "a")], ["x"]),
        (
            [
                nwp.col(
                    [
                        "a",
                        "b",
                        "c",
                        "d",
                        "e",
                        "f",
                        "g",
                        "h",
                        "FIVE",
                        "i",
                        "j",
                        "k",
                        "l",
                        "m",
                        "n",
                        "o",
                        "p",
                        "q",
                        "r",
                        "s",
                        "u",
                    ]
                )
            ],
            ["FIVE"],
        ),
        (
            [nwp.col("a").min().over("c").alias("y"), nwp.col("one").alias("b").last()],
            ["one"],
        ),
        ([nwp.col("a").sort_by("b", "who").alias("f")], ["who"]),
        (
            [
                nwp.nth(0, 5)
                .cast(nw.Int64())
                .abs()
                .cum_sum()
                .over("X", "O", "h", "m", "r", "zee"),
                nwp.col("d", "j"),
                "n",
            ],
            ["O", "X", "zee"],
        ),
    ],
)
def test_prepare_projection_column_not_found(
    into_exprs: IntoExpr | Sequence[IntoExpr], missing: Sequence[str], df_1: Frame
) -> None:
    pattern = re.compile(rf"not found: {re.escape(repr(missing))}")
    with pytest.raises(ColumnNotFoundError, match=pattern):
        df_1.project(into_exprs)


@pytest.mark.parametrize(
    "into_exprs",
    [
        ("a", "b", "c"),
        (["a", "b", "c"]),
        ("a", "b", nwp.col("c")),
        (nwp.col("a"), "b", "c"),
        (nwp.col("a", "b"), "c"),
        ("a", nwp.col("b", "c")),
        ((nwp.nth(0), nwp.nth(1, 2))),
        *MULTI_OUTPUT_EXPRS,
    ],
)
@pytest.mark.parametrize(
    "function",
    [
        nwp.all_horizontal,
        nwp.any_horizontal,
        nwp.sum_horizontal,
        nwp.min_horizontal,
        nwp.max_horizontal,
        nwp.mean_horizontal,
        nwp.concat_str,
    ],
)
def test_prepare_projection_horizontal_alias(
    into_exprs: IntoExpr | Iterable[IntoExpr],
    function: Callable[..., nwp.Expr],
    df_1: Frame,
) -> None:
    # NOTE: See https://github.com/narwhals-dev/narwhals/pull/2572#discussion_r2139965411
    alias_1 = function(into_exprs).alias("alias(x1)")
    out_irs = df_1.project(alias_1)
    assert len(out_irs) == 1
    assert out_irs[0] == named_ir("alias(x1)", function("a", "b", "c"))

    alias_2 = alias_1.alias("alias(x2)")
    out_irs = df_1.project(alias_2)
    assert len(out_irs) == 1
    assert out_irs[0] == named_ir("alias(x2)", function("a", "b", "c"))


@pytest.mark.parametrize(
    "into_exprs", [nwp.nth(-21), nwp.nth(-1, 2, 54, 0), nwp.nth(20), nwp.nth([-10, -100])]
)
def test_prepare_projection_index_error(
    into_exprs: IntoExpr | Iterable[IntoExpr], df_1: Frame
) -> None:
    with pytest.raises(
        ColumnNotFoundError,
        match=re_compile(
            r"invalid.+column.+index.+schema.+last.+column.+`nth\(\-?\d{2,3}\)`"
        ),
    ):
        df_1.project(into_exprs)


@pytest.mark.xfail(
    reason="TODO: binary_expr_combination", raises=MultiOutputExpressionError
)
def test_expand_binary_expr_combination(df_1: Frame) -> None:  # pragma: no cover
    three_to_three = nwp.nth(range(3)) * nwp.nth(3, 4, 5).max()

    expecteds = [
        named_ir("a", nwp.col("a") * nwp.col("d")),
        named_ir("b", nwp.col("b") * nwp.col("e")),
        named_ir("c", nwp.col("c") * nwp.col("f")),
    ]
    actuals = df_1.project(three_to_three)
    for actual, expected in zip_strict(actuals, expecteds):
        assert_expr_ir_equal(actual, expected)


@pytest.mark.xfail(reason="TODO: Move fancy error message", raises=AssertionError)
def test_expand_binary_expr_combination_invalid(df_1: Frame) -> None:  # pragma: no cover
    pattern = re.escape(
        "ncs.all() + ncs.by_name('b', 'c', require_all=True)\n"
        "            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
    )
    all_to_two = nwp.all() + nwp.col("b", "c")
    with pytest.raises(MultiOutputExpressionError, match=pattern):
        df_1.project(all_to_two)

    pattern = re.escape(
        "ncs.by_name('a', 'b', 'c', require_all=True).abs().fill_null([lit(int: 0)]).round() * ncs.by_index([9, 10], require_all=True).cast(Int64).sort(asc)\n"
        "                                                                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
    )
    three_to_two = (
        nwp.col("a", "b", "c").abs().fill_null(0).round(2)
        * nwp.nth(9, 10).cast(nw.Int64).sort()
    )
    with pytest.raises(MultiOutputExpressionError, match=pattern):
        df_1.project(three_to_two)
