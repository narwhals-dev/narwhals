from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable

import pytest

import narwhals as nw
from narwhals import _plan as nwp
from narwhals._plan import expressions as ir, selectors as ndcs
from narwhals._plan._expansion import (
    prepare_projection,
    replace_selector,
    rewrite_special_aliases,
)
from narwhals._plan._parse import parse_into_seq_of_expr_ir
from narwhals._plan.schema import freeze_schema
from narwhals.exceptions import ColumnNotFoundError, ComputeError, DuplicateError
from tests.plan.utils import assert_expr_ir_equal

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from narwhals._plan.typing import IntoExpr, MapIR
    from narwhals.dtypes import DType


@pytest.fixture
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


MULTI_OUTPUT_EXPRS = (
    pytest.param(nwp.col("a", "b", "c")),
    pytest.param(ndcs.numeric() - ndcs.matches("[d-j]")),
    pytest.param(nwp.nth(0, 1, 2)),
    pytest.param(ndcs.by_dtype(nw.Int64, nw.Int32, nw.Int16)),
    pytest.param(ndcs.by_name("a", "b", "c")),
)
"""All of these resolve to `["a", "b", "c"]`."""

BIG_EXCLUDE = ("k", "l", "m", "n", "o", "p", "s", "u", "r", "a", "b", "e", "q")


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
        (
            (
                nwp.col("start")
                .alias("next")
                .sort()
                .round()
                .fill_null(5)
                .alias("noise")
                .name.suffix("_end")
            ),
            "start_end",
        ),
    ],
)
def test_rewrite_special_aliases_single(expr: nwp.Expr, expected: str) -> None:
    # NOTE: We can't use `output_name()` without resolving these rewrites
    # Once they're done, `output_name()` just peeks into `Alias(name=...)`
    ir_input = expr._ir
    with pytest.raises(ComputeError):
        ir_input.meta.output_name()

    ir_output = rewrite_special_aliases(ir_input)
    assert ir_input != ir_output
    actual = ir_output.meta.output_name()
    assert actual == expected


def alias_replace_guarded(name: str) -> MapIR:  # pragma: no cover
    """Guards against repeatedly creating the same alias."""

    def fn(e_ir: ir.ExprIR) -> ir.ExprIR:
        if isinstance(e_ir, ir.Alias) and e_ir.name != name:
            return ir.Alias(expr=e_ir.expr, name=name)
        return e_ir

    return fn


def alias_replace_unguarded(name: str) -> MapIR:  # pragma: no cover
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


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nwp.col("a"), nwp.col("a")),
        (nwp.col("a").max().alias("z"), nwp.col("a").max().alias("z")),
        (ndcs.string(), ir.Columns(names=("k",))),
        (
            ndcs.by_dtype(nw.Datetime("ms"), nw.Date, nw.List(nw.String)),
            nwp.col("n", "s"),
        ),
        (ndcs.string() | ndcs.boolean(), nwp.col("k", "m")),
        (
            ~(ndcs.numeric() | ndcs.string()),
            nwp.col("l", "m", "n", "o", "p", "q", "r", "s", "u"),
        ),
        (
            (
                ndcs.all()
                - (ndcs.categorical() | ndcs.by_name("a", "b") | ndcs.matches("[fqohim]"))
                ^ ndcs.by_name("u", "a", "b", "d", "e", "f", "g")
            ).name.suffix("_after"),
            nwp.col("a", "b", "c", "f", "j", "k", "l", "n", "r", "s").name.suffix(
                "_after"
            ),
        ),
        (
            (ndcs.matches("[a-m]") & ~ndcs.numeric()).sort(nulls_last=True).first()
            != nwp.lit(None),
            nwp.col("k", "l", "m").sort(nulls_last=True).first() != nwp.lit(None),
        ),
        (
            (
                ndcs.numeric()
                .mean()
                .over("k", order_by=ndcs.by_dtype(nw.Date()) | ndcs.boolean())
            ),
            (
                nwp.col("a", "b", "c", "d", "e", "f", "g", "h", "i", "j")
                .mean()
                .over(nwp.col("k"), order_by=nwp.col("m", "n"))
            ),
        ),
        (
            (
                ndcs.datetime()
                .dt.timestamp()
                .min()
                .over(ndcs.string() | ndcs.boolean())
                .last()
                .name.to_uppercase()
            ),
            (
                nwp.col("l", "o")
                .dt.timestamp("us")
                .min()
                .over(nwp.col("k", "m"))
                .last()
                .name.to_uppercase()
            ),
        ),
    ],
)
def test_replace_selector(
    expr: nwp.Selector | nwp.Expr,
    expected: nwp.Expr | ir.ExprIR,
    schema_1: dict[str, DType],
) -> None:
    actual = replace_selector(expr._ir, schema=freeze_schema(**schema_1))
    assert_expr_ir_equal(actual, expected)


@pytest.mark.parametrize(
    ("into_exprs", "expected"),
    [
        ("a", [nwp.col("a")]),
        (nwp.col("b", "c", "d"), [nwp.col("b"), nwp.col("c"), nwp.col("d")]),
        (nwp.nth(6), [nwp.col("g")]),
        (nwp.nth(9, 8, -5), [nwp.col("j"), nwp.col("i"), nwp.col("p")]),
        (
            [nwp.nth(2).alias("c again"), nwp.nth(-1, -2).name.to_uppercase()],
            [
                nwp.col("c").alias("c again"),
                nwp.col("u").alias("U"),
                nwp.col("s").alias("S"),
            ],
        ),
        (
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
        ),
        (
            (ndcs.numeric() - ndcs.by_dtype(nw.Float32(), nw.Float64()))
            .cast(nw.Int64)
            .mean()
            .name.suffix("_mean"),
            [
                nwp.col("a").cast(nw.Int64()).mean().alias("a_mean"),
                nwp.col("b").cast(nw.Int64()).mean().alias("b_mean"),
                nwp.col("c").cast(nw.Int64()).mean().alias("c_mean"),
                nwp.col("d").cast(nw.Int64()).mean().alias("d_mean"),
                nwp.col("e").cast(nw.Int64()).mean().alias("e_mean"),
                nwp.col("f").cast(nw.Int64()).mean().alias("f_mean"),
                nwp.col("g").cast(nw.Int64()).mean().alias("g_mean"),
                nwp.col("h").cast(nw.Int64()).mean().alias("h_mean"),
            ],
        ),
        (
            nwp.col("u").alias("1").alias("2").alias("3").alias("4").name.keep(),
            # NOTE: Would be nice to rewrite with less intermediate steps
            # but retrieving the root name is enough for now
            [nwp.col("u").alias("1").alias("2").alias("3").alias("4").alias("u")],
        ),
        (
            (
                (ndcs.numeric() ^ (ndcs.matches(r"[abcdg]") | ndcs.by_name("i", "f")))
                * 100
            ).name.suffix("_mult_100"),
            [
                (nwp.col("e") * nwp.lit(100)).alias("e_mult_100"),
                (nwp.col("h") * nwp.lit(100)).alias("h_mult_100"),
                (nwp.col("j") * nwp.lit(100)).alias("j_mult_100"),
            ],
        ),
        (
            ndcs.by_dtype(nw.Duration())
            .dt.total_minutes()
            .name.map(lambda nm: f"total_mins: {nm!r} ?"),
            [nwp.col("q").dt.total_minutes().alias("total_mins: 'q' ?")],
        ),
        (
            nwp.col("f", "g")
            .cast(nw.String)
            .str.starts_with("1")
            .all()
            .name.suffix("_all_starts_with_1"),
            [
                nwp.col("f")
                .cast(nw.String)
                .str.starts_with("1")
                .all()
                .alias("f_all_starts_with_1"),
                nwp.col("g")
                .cast(nw.String)
                .str.starts_with("1")
                .all()
                .alias("g_all_starts_with_1"),
            ],
        ),
        (
            nwp.col("a", "b")
            .first()
            .over("c", "e", order_by="d")
            .name.suffix("_first_over_part_order_1"),
            [
                nwp.col("a")
                .first()
                .over(nwp.col("c"), nwp.col("e"), order_by=[nwp.col("d")])
                .alias("a_first_over_part_order_1"),
                nwp.col("b")
                .first()
                .over(nwp.col("c"), nwp.col("e"), order_by=[nwp.col("d")])
                .alias("b_first_over_part_order_1"),
            ],
        ),
        (
            nwp.exclude(BIG_EXCLUDE),
            [
                nwp.col("c"),
                nwp.col("d"),
                nwp.col("f"),
                nwp.col("g"),
                nwp.col("h"),
                nwp.col("i"),
                nwp.col("j"),
            ],
        ),
        (
            nwp.exclude(BIG_EXCLUDE).name.suffix("_2"),
            [
                nwp.col("c").alias("c_2"),
                nwp.col("d").alias("d_2"),
                nwp.col("f").alias("f_2"),
                nwp.col("g").alias("g_2"),
                nwp.col("h").alias("h_2"),
                nwp.col("i").alias("i_2"),
                nwp.col("j").alias("j_2"),
            ],
        ),
        (
            nwp.col("c").alias("c_min_over_order_by").min().over(order_by=ndcs.string()),
            [
                nwp.col("c")
                .alias("c_min_over_order_by")
                .min()
                .over(order_by=[nwp.col("k")])
            ],
        ),
        pytest.param(
            (ndcs.by_name("a", "b", "c") / nwp.col("e").first())
            .over("g", "f", order_by="f")
            .name.prefix("hi_"),
            [
                (nwp.col("a") / nwp.col("e").first())
                .over("g", "f", order_by="f")
                .alias("hi_a"),
                (nwp.col("b") / nwp.col("e").first())
                .over("g", "f", order_by="f")
                .alias("hi_b"),
                (nwp.col("c") / nwp.col("e").first())
                .over("g", "f", order_by="f")
                .alias("hi_c"),
            ],
            id="Selector-BinaryExpr-Over-Prefix",
        ),
    ],
)
def test_prepare_projection(
    into_exprs: IntoExpr | Sequence[IntoExpr],
    expected: Sequence[nwp.Expr],
    schema_1: dict[str, DType],
) -> None:
    irs_in = parse_into_seq_of_expr_ir(into_exprs)
    actual, _, _ = prepare_projection(irs_in, schema_1)
    assert len(actual) == len(expected)
    for lhs, rhs in zip(actual, expected):
        assert_expr_ir_equal(lhs, rhs)


@pytest.mark.parametrize(
    "expr",
    [
        nwp.all(),
        nwp.nth(1, 2, 3),
        nwp.col("a", "b", "c"),
        ndcs.boolean() | ndcs.categorical(),
        (ndcs.by_name("a", "b") | ndcs.string()),
        (nwp.col("b", "c") & nwp.col("a")),
        nwp.col("a", "b").min().over("c", order_by="e"),
        (~ndcs.by_dtype(nw.Int64()) - ndcs.datetime()),
        nwp.nth(6, 2).abs().cast(nw.Int32()) + 10,
        *MULTI_OUTPUT_EXPRS,
    ],
)
def test_prepare_projection_duplicate(expr: nwp.Expr, schema_1: dict[str, DType]) -> None:
    irs = parse_into_seq_of_expr_ir(expr.alias("dupe"))
    pattern = re.compile(r"\.alias\(.dupe.\)")
    with pytest.raises(DuplicateError, match=pattern):
        prepare_projection(irs, schema_1)


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
    into_exprs: IntoExpr | Sequence[IntoExpr],
    missing: Sequence[str],
    schema_1: dict[str, DType],
) -> None:
    pattern = re.compile(rf"not found: {re.escape(repr(missing))}")
    irs = parse_into_seq_of_expr_ir(into_exprs)
    with pytest.raises(ColumnNotFoundError, match=pattern):
        prepare_projection(irs, schema_1)


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
    schema_1: dict[str, DType],
) -> None:
    # NOTE: See https://github.com/narwhals-dev/narwhals/pull/2572#discussion_r2139965411
    expr = function(into_exprs)
    alias_1 = expr.alias("alias(x1)")
    irs = parse_into_seq_of_expr_ir(alias_1)
    out_irs, _, _ = prepare_projection(irs, schema_1)
    assert len(out_irs) == 1
    assert out_irs[0] == function("a", "b", "c").alias("alias(x1)")._ir

    alias_2 = alias_1.alias("alias(x2)")
    irs = parse_into_seq_of_expr_ir(alias_2)
    out_irs, _, _ = prepare_projection(irs, schema_1)
    assert len(out_irs) == 1
    assert out_irs[0] == function("a", "b", "c").alias("alias(x1)").alias("alias(x2)")._ir


@pytest.mark.parametrize(
    "into_exprs", [nwp.nth(-21), nwp.nth(-1, 2, 54, 0), nwp.nth(20), nwp.nth([-10, -100])]
)
def test_prepare_projection_index_error(
    into_exprs: IntoExpr | Iterable[IntoExpr], schema_1: dict[str, DType]
) -> None:
    irs = parse_into_seq_of_expr_ir(into_exprs)
    pattern = re.compile(r"invalid.+index.+nth", re.DOTALL | re.IGNORECASE)
    with pytest.raises(ComputeError, match=pattern):
        prepare_projection(irs, schema_1)
