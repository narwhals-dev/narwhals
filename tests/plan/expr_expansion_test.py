from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable

import pytest

import narwhals as nw
from narwhals._plan import demo as nwd, selectors as ndcs
from narwhals._plan.expr import Alias, Columns
from narwhals._plan.expr_expansion import (
    freeze_schema,
    prepare_projection,
    replace_selector,
    rewrite_special_aliases,
)
from narwhals._plan.expr_parsing import parse_into_seq_of_expr_ir
from narwhals.exceptions import ColumnNotFoundError, ComputeError, DuplicateError
from tests.plan.utils import assert_expr_ir_equal

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from narwhals._plan.common import ExprIR
    from narwhals._plan.dummy import DummyExpr, DummySelector
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
    pytest.param(nwd.col("a", "b", "c")),
    pytest.param(ndcs.numeric() - ndcs.matches("[d-j]")),
    pytest.param(nwd.nth(0, 1, 2)),
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
        (nwd.col("a").name.to_uppercase(), "A"),
        (nwd.col("B").name.to_lowercase(), "b"),
        (nwd.col("c").name.suffix("_after"), "c_after"),
        (nwd.col("d").name.prefix("before_"), "before_d"),
        (
            nwd.col("aBcD EFg hi").name.map(udf_name_map),
            "original='aBcD EFg hi' | upper='ABCD EFG HI' | lower='abcd efg hi' | title='Abcd Efg Hi'",
        ),
        (nwd.col("a").min().alias("b").over("c").alias("d").max().name.keep(), "a"),
        (
            (
                nwd.col("hello")
                .sort_by(nwd.col("ignore me"))
                .max()
                .over("ignore me as well")
                .first()
                .name.to_uppercase()
            ),
            "HELLO",
        ),
        (
            (
                nwd.col("start")
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
def test_rewrite_special_aliases_single(expr: DummyExpr, expected: str) -> None:
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

    def fn(ir: ExprIR) -> ExprIR:
        if isinstance(ir, Alias) and ir.name != name:
            return Alias(expr=ir.expr, name=name)
        return ir

    return fn


def alias_replace_unguarded(name: str) -> MapIR:  # pragma: no cover
    """**Does not guard against recursion**!

    Handling the recursion stopping **should be** part of the impl of `ExprIR.map_ir`.

    - *Ideally*, return an identical result to `alias_replace_guarded` (after the same number of iterations)
    - *Pragmatically*, it might require an extra iteration to detect a cycle
    """

    def fn(ir: ExprIR) -> ExprIR:
        if isinstance(ir, Alias):
            return Alias(expr=ir.expr, name=name)
        return ir

    return fn


@pytest.mark.parametrize(
    ("expr", "function", "expected"),
    [
        (nwd.col("a"), alias_replace_guarded("never"), nwd.col("a")),
        (nwd.col("a"), alias_replace_unguarded("never"), nwd.col("a")),
        (nwd.col("a").alias("b"), alias_replace_guarded("c"), nwd.col("a").alias("c")),
        (nwd.col("a").alias("b"), alias_replace_unguarded("c"), nwd.col("a").alias("c")),
        (
            nwd.col("a").alias("d").first().over("b", order_by="c").alias("e"),
            alias_replace_guarded("d"),
            nwd.col("a").alias("d").first().over("b", order_by="c").alias("d"),
        ),
        (
            nwd.col("a").alias("d").first().over("b", order_by="c").alias("e"),
            alias_replace_unguarded("d"),
            nwd.col("a").alias("d").first().over("b", order_by="c").alias("d"),
        ),
        (
            nwd.col("a").alias("e").abs().alias("f").sort().alias("g"),
            alias_replace_guarded("e"),
            nwd.col("a").alias("e").abs().alias("e").sort().alias("e"),
        ),
        (
            nwd.col("a").alias("e").abs().alias("f").sort().alias("g"),
            alias_replace_unguarded("e"),
            nwd.col("a").alias("e").abs().alias("e").sort().alias("e"),
        ),
    ],
)
def test_map_ir_recursive(expr: DummyExpr, function: MapIR, expected: DummyExpr) -> None:
    actual = expr._ir.map_ir(function)
    assert_expr_ir_equal(actual, expected)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nwd.col("a"), nwd.col("a")),
        (nwd.col("a").max().alias("z"), nwd.col("a").max().alias("z")),
        (ndcs.string(), Columns(names=("k",))),
        (
            ndcs.by_dtype(nw.Datetime("ms"), nw.Date, nw.List(nw.String)),
            nwd.col("n", "s"),
        ),
        (ndcs.string() | ndcs.boolean(), nwd.col("k", "m")),
        (
            ~(ndcs.numeric() | ndcs.string()),
            nwd.col("l", "m", "n", "o", "p", "q", "r", "s", "u"),
        ),
        (
            (
                ndcs.all()
                - (ndcs.categorical() | ndcs.by_name("a", "b") | ndcs.matches("[fqohim]"))
                ^ ndcs.by_name("u", "a", "b", "d", "e", "f", "g")
            ).name.suffix("_after"),
            nwd.col("a", "b", "c", "f", "j", "k", "l", "n", "r", "s").name.suffix(
                "_after"
            ),
        ),
        (
            (ndcs.matches("[a-m]") & ~ndcs.numeric()).sort(nulls_last=True).first()
            != nwd.lit(None),
            nwd.col("k", "l", "m").sort(nulls_last=True).first() != nwd.lit(None),
        ),
        (
            (
                ndcs.numeric()
                .mean()
                .over("k", order_by=ndcs.by_dtype(nw.Date()) | ndcs.boolean())
            ),
            (
                nwd.col("a", "b", "c", "d", "e", "f", "g", "h", "i", "j")
                .mean()
                .over(nwd.col("k"), order_by=nwd.col("m", "n"))
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
                nwd.col("l", "o")
                .dt.timestamp("us")
                .min()
                .over(nwd.col("k", "m"))
                .last()
                .name.to_uppercase()
            ),
        ),
    ],
)
def test_replace_selector(
    expr: DummySelector | DummyExpr,
    expected: DummyExpr | ExprIR,
    schema_1: dict[str, DType],
) -> None:
    group_by_keys = ()
    actual = replace_selector(expr._ir, group_by_keys, schema=freeze_schema(**schema_1))
    assert_expr_ir_equal(actual, expected)


@pytest.mark.parametrize(
    ("into_exprs", "expected"),
    [
        ("a", [nwd.col("a")]),
        (nwd.col("b", "c", "d"), [nwd.col("b"), nwd.col("c"), nwd.col("d")]),
        (nwd.nth(6), [nwd.col("g")]),
        (nwd.nth(9, 8, -5), [nwd.col("j"), nwd.col("i"), nwd.col("p")]),
        (
            [nwd.nth(2).alias("c again"), nwd.nth(-1, -2).name.to_uppercase()],
            [
                nwd.col("c").alias("c again"),
                nwd.col("u").alias("U"),
                nwd.col("s").alias("S"),
            ],
        ),
        (
            nwd.all(),
            [
                nwd.col("a"),
                nwd.col("b"),
                nwd.col("c"),
                nwd.col("d"),
                nwd.col("e"),
                nwd.col("f"),
                nwd.col("g"),
                nwd.col("h"),
                nwd.col("i"),
                nwd.col("j"),
                nwd.col("k"),
                nwd.col("l"),
                nwd.col("m"),
                nwd.col("n"),
                nwd.col("o"),
                nwd.col("p"),
                nwd.col("q"),
                nwd.col("r"),
                nwd.col("s"),
                nwd.col("u"),
            ],
        ),
        (
            (ndcs.numeric() - ndcs.by_dtype(nw.Float32(), nw.Float64()))
            .cast(nw.Int64())
            .mean()
            .name.suffix("_mean"),
            [
                nwd.col("a").cast(nw.Int64()).mean().alias("a_mean"),
                nwd.col("b").cast(nw.Int64()).mean().alias("b_mean"),
                nwd.col("c").cast(nw.Int64()).mean().alias("c_mean"),
                nwd.col("d").cast(nw.Int64()).mean().alias("d_mean"),
                nwd.col("e").cast(nw.Int64()).mean().alias("e_mean"),
                nwd.col("f").cast(nw.Int64()).mean().alias("f_mean"),
                nwd.col("g").cast(nw.Int64()).mean().alias("g_mean"),
                nwd.col("h").cast(nw.Int64()).mean().alias("h_mean"),
            ],
        ),
        (
            nwd.col("u").alias("1").alias("2").alias("3").alias("4").name.keep(),
            # NOTE: Would be nice to rewrite with less intermediate steps
            # but retrieving the root name is enough for now
            [nwd.col("u").alias("1").alias("2").alias("3").alias("4").alias("u")],
        ),
        (
            (
                (ndcs.numeric() ^ (ndcs.matches(r"[abcdg]") | ndcs.by_name("i", "f")))
                * 100
            ).name.suffix("_mult_100"),
            [
                (nwd.col("e") * nwd.lit(100)).alias("e_mult_100"),
                (nwd.col("h") * nwd.lit(100)).alias("h_mult_100"),
                (nwd.col("j") * nwd.lit(100)).alias("j_mult_100"),
            ],
        ),
        (
            ndcs.by_dtype(nw.Duration())
            .dt.total_minutes()
            .name.map(lambda nm: f"total_mins: {nm!r} ?"),
            [nwd.col("q").dt.total_minutes().alias("total_mins: 'q' ?")],
        ),
        (
            nwd.col("f", "g")
            .cast(nw.String())
            .str.starts_with("1")
            .all()
            .name.suffix("_all_starts_with_1"),
            [
                nwd.col("f")
                .cast(nw.String())
                .str.starts_with("1")
                .all()
                .alias("f_all_starts_with_1"),
                nwd.col("g")
                .cast(nw.String())
                .str.starts_with("1")
                .all()
                .alias("g_all_starts_with_1"),
            ],
        ),
        (
            nwd.col("a", "b")
            .first()
            .over("c", "e", order_by="d")
            .name.suffix("_first_over_part_order_1"),
            [
                nwd.col("a")
                .first()
                .over(nwd.col("c"), nwd.col("e"), order_by=[nwd.col("d")])
                .alias("a_first_over_part_order_1"),
                nwd.col("b")
                .first()
                .over(nwd.col("c"), nwd.col("e"), order_by=[nwd.col("d")])
                .alias("b_first_over_part_order_1"),
            ],
        ),
        (
            nwd.exclude(BIG_EXCLUDE),
            [
                nwd.col("c"),
                nwd.col("d"),
                nwd.col("f"),
                nwd.col("g"),
                nwd.col("h"),
                nwd.col("i"),
                nwd.col("j"),
            ],
        ),
        (
            nwd.exclude(BIG_EXCLUDE).name.suffix("_2"),
            [
                nwd.col("c").alias("c_2"),
                nwd.col("d").alias("d_2"),
                nwd.col("f").alias("f_2"),
                nwd.col("g").alias("g_2"),
                nwd.col("h").alias("h_2"),
                nwd.col("i").alias("i_2"),
                nwd.col("j").alias("j_2"),
            ],
        ),
        (
            nwd.col("c").alias("c_min_over_order_by").min().over(order_by=ndcs.string()),
            [
                nwd.col("c")
                .alias("c_min_over_order_by")
                .min()
                .over(order_by=[nwd.col("k")])
            ],
        ),
    ],
)
def test_prepare_projection(
    into_exprs: IntoExpr | Sequence[IntoExpr],
    expected: Sequence[DummyExpr],
    schema_1: dict[str, DType],
) -> None:
    irs_in = parse_into_seq_of_expr_ir(into_exprs)
    actual, _ = prepare_projection(irs_in, schema_1)
    assert len(actual) == len(expected)
    for lhs, rhs in zip(actual, expected):
        assert_expr_ir_equal(lhs, rhs)


@pytest.mark.parametrize(
    "expr",
    [
        nwd.all(),
        nwd.nth(1, 2, 3),
        nwd.col("a", "b", "c"),
        ndcs.boolean() | ndcs.categorical(),
        (ndcs.by_name("a", "b") | ndcs.string()),
        (nwd.col("b", "c") & nwd.col("a")),
        nwd.col("a", "b").min().over("c", order_by="e"),
        (~ndcs.by_dtype(nw.Int64()) - ndcs.datetime()),
        nwd.nth(6, 2).abs().cast(nw.Int32()) + 10,
        *MULTI_OUTPUT_EXPRS,
    ],
)
def test_prepare_projection_duplicate(
    expr: DummyExpr, schema_1: dict[str, DType]
) -> None:
    irs = parse_into_seq_of_expr_ir(expr.alias("dupe"))
    pattern = re.compile(r"\.alias\(.dupe.\)")
    with pytest.raises(DuplicateError, match=pattern):
        prepare_projection(irs, schema_1)


@pytest.mark.parametrize(
    ("into_exprs", "missing"),
    [
        ([nwd.col("y", "z")], ["y", "z"]),
        ([nwd.col("a", "b", "z")], ["z"]),
        ([nwd.col("x", "b", "a")], ["x"]),
        (
            [
                nwd.col(
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
            [nwd.col("a").min().over("c").alias("y"), nwd.col("one").alias("b").last()],
            ["one"],
        ),
        ([nwd.col("a").sort_by("b", "who").alias("f")], ["who"]),
        (
            [
                nwd.nth(0, 5)
                .cast(nw.Int64())
                .abs()
                .cum_sum()
                .over("X", "O", "h", "m", "r", "zee"),
                nwd.col("d", "j"),
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
        ("a", "b", nwd.col("c")),
        (nwd.col("a"), "b", "c"),
        (nwd.col("a", "b"), "c"),
        ("a", nwd.col("b", "c")),
        ((nwd.nth(0), nwd.nth(1, 2))),
        *MULTI_OUTPUT_EXPRS,
    ],
)
@pytest.mark.parametrize(
    "function",
    [
        nwd.all_horizontal,
        nwd.any_horizontal,
        nwd.sum_horizontal,
        nwd.min_horizontal,
        nwd.max_horizontal,
        nwd.mean_horizontal,
        nwd.concat_str,
    ],
)
def test_prepare_projection_horizontal_alias(
    into_exprs: IntoExpr | Iterable[IntoExpr],
    function: Callable[..., DummyExpr],
    schema_1: dict[str, DType],
) -> None:
    # NOTE: See https://github.com/narwhals-dev/narwhals/pull/2572#discussion_r2139965411
    expr = function(into_exprs)
    alias_1 = expr.alias("alias(x1)")
    irs = parse_into_seq_of_expr_ir(alias_1)
    out_irs, _ = prepare_projection(irs, schema_1)
    assert len(out_irs) == 1
    assert out_irs[0] == function("a", "b", "c").alias("alias(x1)")._ir

    alias_2 = alias_1.alias("alias(x2)")
    irs = parse_into_seq_of_expr_ir(alias_2)
    out_irs, _ = prepare_projection(irs, schema_1)
    assert len(out_irs) == 1
    assert out_irs[0] == function("a", "b", "c").alias("alias(x1)").alias("alias(x2)")._ir


@pytest.mark.parametrize(
    "into_exprs", [nwd.nth(-21), nwd.nth(-1, 2, 54, 0), nwd.nth(20), nwd.nth([-10, -100])]
)
def test_prepare_projection_index_error(
    into_exprs: IntoExpr | Iterable[IntoExpr], schema_1: dict[str, DType]
) -> None:
    irs = parse_into_seq_of_expr_ir(into_exprs)
    pattern = re.compile(r"invalid.+index.+nth", re.DOTALL | re.IGNORECASE)
    with pytest.raises(ComputeError, match=pattern):
        prepare_projection(irs, schema_1)
