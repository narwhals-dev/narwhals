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
    InvalidOperationError,
    MultiOutputExpressionError,
)
from tests.plan.utils import Frame, assert_expr_ir_equal, cols, named_ir, re_compile

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
    nwp.col("a", "b", "c"),
    ncs.numeric() - ncs.matches("[d-j]"),
    nwp.nth(0, 1, 2),
    ncs.by_dtype(nw.Int64, nw.Int32, nw.Int16),
    ncs.by_name("a", "b", "c"),
)
"""All of these resolve to `["a", "b", "c"]`."""

BIG_EXCLUDE = nwp.exclude("k", "l", "m", "n", "o", "p", "s", "u", "r", "a", "b", "e", "q")

I64 = nw.Int64()
STR = nw.String()


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


def test_keep_name_no_names(df_1: Frame) -> None:
    with pytest.raises(
        InvalidOperationError,
        match=r"name.keep` expected at least one.+name.+got.+lit.+1.+name.keep",
    ):
        df_1.project(nwp.lit(1).name.keep())


def alias_replace_guarded(name: str) -> MapIR:
    """Guards against repeatedly creating the same alias."""

    def fn(e_ir: ir.ExprIR) -> ir.ExprIR:
        if isinstance(e_ir, ir.Alias) and e_ir.name != name:
            return ir.Alias(expr=e_ir.expr, name=name)
        return e_ir

    return fn


def alias_replace_unguarded(name: str) -> MapIR:
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
        pytest.param(nwp.col("b", "c", "d"), list(cols("b", "c", "d")), id="ByName(3)"),
        pytest.param(nwp.nth(6), [nwp.col("g")], id="ByIndex(1)"),
        pytest.param(nwp.nth(9, 8, -5), list(cols("j", "i", "p")), id="ByIndex(3)"),
        pytest.param(
            [nwp.nth(2).alias("c again"), nwp.nth(-1, -2).name.to_uppercase()],
            [
                named_ir("c again", nwp.col("c")),
                named_ir("U", nwp.col("u")),
                named_ir("S", nwp.col("s")),
            ],
            id="ByIndex(1)-Alias-ByIndex(2)-Uppercase",
        ),
        pytest.param(nwp.all(), list(cols(*"abcdefghijklmnopqrsu")), id="All"),
        pytest.param(
            (ncs.numeric() - ncs.float()).cast(nw.Int64).mean().name.suffix("_mean"),
            [named_ir(f"{s}_mean", nwp.col(s).cast(I64).mean()) for s in "abcdefgh"],
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
            ).name.suffix(" * 100"),
            [named_ir(f"{s} * 100", nwp.col(s) * nwp.lit(100)) for s in "ehj"],
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
            .name.suffix("_with"),
            [
                named_ir("f_with", nwp.col("f").cast(STR).str.starts_with("1").all()),
                named_ir("g_with", nwp.col("g").cast(STR).str.starts_with("1").all()),
            ],
            id="ByName(2)-Cast-StartsWith-All-Suffix",
        ),
        pytest.param(
            nwp.col("a", "b").first().over("c", "e", order_by="d").name.suffix("_first"),
            [
                named_ir(
                    f"{s}_first",
                    nwp.col(s).first().over(nwp.col("c"), nwp.col("e"), order_by="d"),
                )
                for s in "ab"
            ],
            id="ByName(2)-First-Over-Partitioned-Ordered-Suffix",
        ),
        pytest.param(
            BIG_EXCLUDE, list(cols("c", "d", "f", "g", "h", "i", "j")), id="Exclude"
        ),
        pytest.param(
            BIG_EXCLUDE.name.suffix("_2"),
            [named_ir(f"{s}_2", nwp.col(s)) for s in "cdfghij"],
            id="Exclude-Suffix",
        ),
        pytest.param(
            nwp.col("c").alias("c_min").min().over(order_by=ncs.string()),
            [named_ir("c_min", nwp.col("c").min().over(order_by=[nwp.col("k")]))],
            id="Alias-Min-Over-Order-By-Selector",
        ),
        pytest.param(
            (ncs.by_name("a", "b", "c") / nwp.col("e").first())
            .over("g", "f", order_by="f")
            .name.prefix("hi_"),
            [
                named_ir(
                    f"hi_{s}",
                    (nwp.col(s) / nwp.col("e").first()).over("g", "f", order_by="f"),
                )
                for s in "abc"
            ],
            id="Selector-BinaryExpr-Over-Prefix",
        ),
        pytest.param(
            [
                nwp.col("c").sort_by(nwp.col("c", "i")).first().alias("ByName(2)"),
                nwp.col("c").sort_by("c", "i").first().alias("Column_x2"),
            ],
            [
                named_ir("ByName(2)", nwp.col("c").sort_by("c", "i").first()),
                named_ir("Column_x2", nwp.col("c").sort_by("c", "i").first()),
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


_very_long = (*"abcdefgh", "FIVE", *"ijklmnopqrsu")


@pytest.mark.parametrize(
    ("into_exprs", "missing"),
    [
        ([nwp.col("y", "z")], ["y", "z"]),
        ([nwp.col("a", "b", "z")], ["z"]),
        ([nwp.col("x", "b", "a")], ["x"]),
        ([nwp.col(_very_long)], ["FIVE"]),
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
    pattern = re_compile(
        r"invalid.+column.+index.+schema.+last.+column.+`nth\(\-?\d{2,3}\)`"
    )
    with pytest.raises(ColumnNotFoundError, match=pattern):
        df_1.project(into_exprs)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (
            nwp.nth(range(3)) * nwp.nth(3, 4, 5).max(),
            [
                nwp.col("a") * nwp.col("d").max(),
                nwp.col("b") * nwp.col("e").max(),
                nwp.col("c") * nwp.col("f").max(),
            ],
        ),
        (
            (10 / nwp.col("e", "d", "b", "a")).name.keep(),
            [named_ir(s, 10 / nwp.col(s)) for s in "edba"],
        ),
        (
            (
                (ncs.categorical() | ncs.string())
                .as_expr()
                .cast(STR)
                .str.len_chars()
                .name.map(lambda s: f"len_chars({s!r})")
                - ncs.by_dtype(nw.UInt16).as_expr()
            ).name.suffix("-col('g')"),
            [
                named_ir(
                    "len_chars('k')-col('g')",
                    nwp.col("k").cast(STR).str.len_chars() - nwp.col("g"),
                ),
                named_ir(
                    "len_chars('p')-col('g')",
                    nwp.col("p").cast(STR).str.len_chars() - nwp.col("g"),
                ),
            ],
        ),
        (
            (nwp.all().first() == nwp.all().last()).name.suffix("_first_eq_last"),
            [
                named_ir(f"{e.meta.output_name()}_first_eq_last", (e.first() == e.last()))
                for e in cols(*"abcdefghijklmnopqrsu")
            ],
        ),
    ],
    ids=["3:3", "1:4", "2:1", "All:All"],
)
def test_expand_binary_expr_combination(
    df_1: Frame, expr: nwp.Expr, expected: Iterable[ir.NamedIR | nwp.Expr]
) -> None:
    actuals = df_1.project(expr)
    for actual, expect in zip_strict(actuals, expected):
        assert_expr_ir_equal(actual, expect)


def test_expand_binary_expr_combination_invalid(df_1: Frame) -> None:
    # fmt: off
    expr = re.escape(
        "ncs.all() + ncs.by_name('b', 'c')\n"
        "^^^^^^^^^"
    )
    # fmt: on
    shapes = "(20 != 2)"
    pattern = rf"{shapes}.+\n{expr}"
    all_to_two = nwp.all() + nwp.col("b", "c")
    with pytest.raises(MultiOutputExpressionError, match=pattern):
        df_1.project(all_to_two)

    expr = re.escape(
        "ncs.by_name('a', 'b').abs().fill_null([lit(int: 0)]).round() * ncs.by_index([9, 10, 11]).cast(Int64).sort(asc)\n"
        "                                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
    )
    shapes = "(2 != 3)"
    pattern = rf"{shapes}.+\n{expr}"
    two_to_three = (
        nwp.col("a", "b").abs().fill_null(0).round(2)
        * nwp.nth(9, 10, 11).cast(nw.Int64).sort()
    )
    with pytest.raises(MultiOutputExpressionError, match=pattern):
        df_1.project(two_to_three)

    # fmt: off
    expr = re.escape(
        "ncs.numeric() / [ncs.numeric() - ncs.by_dtype([Int64])]\n"
        "^^^^^^^^^^^^^"
    )
    # fmt: on
    shapes = "(10 != 9)"
    pattern = rf"{shapes}.+\n{expr}"
    ten_to_nine = (
        ncs.numeric().as_expr() / (ncs.numeric() - ncs.by_dtype(nw.Int64)).as_expr()
    )
    with pytest.raises(MultiOutputExpressionError, match=pattern):
        df_1.project(ten_to_nine)


def test_exclude_when_then_21352() -> None:
    # https://github.com/pola-rs/polars/blob/7fc9f1875714fe9893c4d849b9593c1e4db1e854/py-polars/tests/unit/operations/test_selectors.py#L1090-L1096
    df = Frame.from_mapping({"A": nw.Int64(), "B": nw.Int64()})
    exclude_b = nwp.all().exclude("B")
    combination = nwp.when(nwp.lit(True)).then(nwp.all()).otherwise(exclude_b)
    df.assert_selects(exclude_b, "A")
    actuals = df.project(combination)
    expected = (
        named_ir("A", nwp.when(nwp.lit(True)).then("A").otherwise("A")),
        named_ir("B", nwp.when(nwp.lit(True)).then("B").otherwise("A")),
    )

    for actual, expect in zip_strict(actuals, expected):
        assert_expr_ir_equal(actual, expect)


def test_expand_ternary_expr_combination_1(df_1: Frame) -> None:
    expr = (
        nwp.when(ncs.boolean())
        .then(ncs.matches(r"[ji]"))
        .otherwise(nwp.col("j", "i"))
        .name.to_uppercase()
    )
    expected = (
        # `Matches` doesn't define column order, but `ByName` can
        named_ir("I", nwp.when("m").then("i").otherwise("j")),
        named_ir("J", nwp.when("m").then("j").otherwise("i")),
    )
    actuals = df_1.project(expr)
    for actual, expect in zip_strict(actuals, expected):
        assert_expr_ir_equal(actual, expect)


def test_expand_ternary_expr_combination_2(df_1: Frame) -> None:
    less_than_5_flip = nwp.when(nwp.col("a", "b", "c") <= 5).then(nwp.nth(2, 1, 0))
    expected = (
        named_ir("c", nwp.when(nwp.col("a") <= 5).then("c")),
        named_ir("b", nwp.when(nwp.col("b") <= 5).then("b")),
        named_ir("a", nwp.when(nwp.col("c") <= 5).then("a")),
    )
    actuals = df_1.project(less_than_5_flip)
    for actual, expect in zip_strict(actuals, expected):
        assert_expr_ir_equal(actual, expect)


def test_expand_ternary_expr_combination_3(df_1: Frame) -> None:
    integer = ncs.integer()
    signed_integer = integer - ncs.by_dtype(
        nw.UInt64, nw.UInt32, nw.UInt16, nw.UInt8, nw.UInt128
    )
    unsigned_integer = integer - ncs.by_dtype(
        nw.Int64, nw.Int32, nw.Int16, nw.Int8, nw.Int128
    )
    field = nwp.col("u").struct.field("a")
    explode_brain = (
        nwp.when((unsigned_integer.is_null().as_expr()) & nwp.nth(2).is_not_null())
        .then(signed_integer.name.suffix("_after"))
        .when(ncs.by_dtype(nw.Int16).is_null())
        .then(field)
    )

    c_not_null = nwp.col("c").is_not_null()
    c_is_null = nwp.col("c").is_null()

    def expands(output_name: str, lhs_predicate_name: str, then_name: str) -> ir.NamedIR:
        return named_ir(
            output_name,
            nwp.when(nwp.col(lhs_predicate_name).is_null() & c_not_null)
            .then(then_name)
            .when(c_is_null)
            .then(field),
        )

    expected = (
        expands("a_after", "e", "a"),
        expands("b_after", "f", "b"),
        expands("c_after", "g", "c"),
        expands("d_after", "h", "d"),
    )
    actuals = df_1.project(explode_brain)
    for actual, expect in zip_strict(actuals, expected):
        assert_expr_ir_equal(actual, expect)


def test_expand_ternary_expr_combination_invalid(df_1: Frame) -> None:
    with _raises_when_multi(3, 2, 1):
        df_1.project(nwp.when(nwp.col("a", "b", "c").is_finite()).then(nwp.max("c", "d")))

    with _raises_when_multi(1, 2, 3):
        df_1.project(
            nwp.when(nwp.len() >= 1).then(ncs.float()).otherwise(nwp.nth(1, 2, 3))
        )

    with _raises_when_multi(8, 1, 2):
        df_1.project(
            nwp.when(ncs.integer().cast(nw.Boolean))
            .then(nwp.lit("truthy"))
            .otherwise((ncs.string() | ncs.enum()).cast(nw.String))
        )


def _raises_when_multi(
    predicate: int, truthy: int, falsy: int
) -> pytest.RaisesExc[MultiOutputExpressionError]:
    shapes = f"({predicate} != {truthy} != {falsy})"
    return pytest.raises(MultiOutputExpressionError, match=re.escape(shapes))


def test_expand_function_expr_multi_invalid(df_1: Frame) -> None:
    first_column = re.escape("col('a')")
    last_selected_column = re.escape("col('h')")
    found = rf".+{first_column}.+{last_selected_column}\Z"
    pattern = re_compile(
        rf"not supported.+context.+{first_column}\.is_in.+ncs\.integer.+ncs\.integer.+expanded into 8 outputs{found}"
    )
    with pytest.raises(MultiOutputExpressionError, match=pattern):
        df_1.project(nwp.col("a").is_in(ncs.integer()))
    with pytest.raises(MultiOutputExpressionError, match=r"expanded into 20 outputs"):
        df_1.project(nwp.col("d").is_in(nwp.all()))


def test_over_order_by_names() -> None:
    expr = nwp.col("a").first().over(order_by=ncs.string())
    e_ir = expr._ir
    assert isinstance(e_ir, ir.OverOrdered)
    pattern = re_compile(r"cannot use.+order_by_names.+before.+expansion.+ncs.string\(\)")
    with pytest.raises(InvalidOperationError, match=pattern):
        list(e_ir.order_by_names())
