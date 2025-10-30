from __future__ import annotations

import datetime as dt
import re
import string
from typing import Any

import pytest

import narwhals._plan.selectors as ncs
from narwhals import _plan as nwp
from narwhals.exceptions import ComputeError
from tests.plan.utils import series
from tests.utils import POLARS_VERSION

pytest.importorskip("polars")
import polars as pl

if POLARS_VERSION >= (1, 0):
    # https://github.com/pola-rs/polars/pull/16743
    OVER_CASE = (
        nwp.col("a").last().over("b", order_by="c"),
        pl.col("a").last().over("b", order_by="c"),
        ["a", "b"],
    )
else:  # pragma: no cover
    OVER_CASE = (nwp.col("a").last().over("b"), pl.col("a").last().over("b"), ["a", "b"])
if POLARS_VERSION >= (0, 20, 5):
    LEN_CASE = (nwp.len(), pl.len(), "len")
else:  # pragma: no cover
    LEN_CASE = (nwp.len().alias("count"), pl.count(), "count")


XFAIL_LITERAL_LIST = pytest.mark.xfail(
    reason="'list' is not supported in `nw.lit`", raises=TypeError
)


@pytest.mark.parametrize(
    ("nw_expr", "pl_expr", "expected"),
    [
        (
            nwp.col("a").alias("b").min().alias("c").alias("d"),
            pl.col("a").alias("b").min().alias("c").alias("d"),
            ["a"],
        ),
        (
            (nwp.col("a") + (nwp.col("a") - nwp.col("b"))).alias("c"),
            (pl.col("a") + (pl.col("a") - pl.col("b"))).alias("c"),
            ["a", "a", "b"],
        ),
        OVER_CASE,
        (
            (nwp.col("a", "b", "c").sort().abs() * 20).max(),
            (pl.col("a", "b", "c").sort().abs() * 20).max(),
            [],
        ),
        (nwp.all().mean(), pl.all().mean(), []),
        (nwp.all().mean().sort_by("d"), pl.all().mean().sort_by("d"), ["d"]),
        (
            nwp.all_horizontal(*string.ascii_letters),
            pl.all_horizontal(*string.ascii_letters),
            list(string.ascii_letters),
        ),
    ],
)
def test_meta_root_names(
    nw_expr: nwp.Expr, pl_expr: pl.Expr, expected: list[str]
) -> None:
    pl_result = pl_expr.meta.root_names()
    nw_result = nw_expr.meta.root_names()
    assert nw_result == expected
    assert nw_result == pl_result


@pytest.mark.parametrize(
    ("nw_expr", "pl_expr", "expected"),
    [
        (nwp.col("a"), pl.col("a"), "a"),
        (nwp.lit(1), pl.lit(1), "literal"),
        LEN_CASE,
        pytest.param(
            (
                nwp.col("a")
                .alias("b")
                .min()
                .alias("c")
                .over("e", "f")
                .sort_by(nwp.col("i"), nwp.col("g", "h"))
            ),
            (
                pl.col("a")
                .alias("b")
                .min()
                .alias("c")
                .over("e", "f")
                .sort_by(pl.col("i"), pl.col("g", "h"))
            ),
            "c",
            id="Kitchen-Sink",
        ),
        pytest.param(
            nwp.col("c").alias("x").fill_null(50),
            pl.col("c").alias("x").fill_null(50),
            "x",
            id="FunctionExpr-Literal",
        ),
        pytest.param(
            (
                nwp.col("ROOT")
                .alias("ROOT-ALIAS")
                .filter(nwp.col("b") >= 30, nwp.col("c").alias("d") == 7)
                + nwp.col("RHS").alias("RHS-ALIAS")
            ),
            (
                pl.col("ROOT")
                .alias("ROOT-ALIAS")
                .filter(pl.col("b") >= 30, pl.col("c").alias("d") == 7)
                + pl.col("RHS").alias("RHS-ALIAS")
            ),
            "ROOT-ALIAS",
            id="BinaryExpr-Multiple",
        ),
        pytest.param(
            nwp.col("ROOT").alias("ROOT-ALIAS").mean().over(nwp.col("a").alias("b")),
            pl.col("ROOT").alias("ROOT-ALIAS").mean().over(pl.col("a").alias("b")),
            "ROOT-ALIAS",
            id="WindowExpr",
        ),
        pytest.param(
            nwp.when(nwp.col("a").alias("a?")).then(10),
            pl.when(pl.col("a").alias("a?")).then(10),
            "literal",
            id="When-Literal",
        ),
        pytest.param(
            nwp.when(nwp.col("a").alias("a?")).then(nwp.col("b")).otherwise(20),
            pl.when(pl.col("a").alias("a?")).then(pl.col("b")).otherwise(20),
            "b",
            id="When-Column-Literal",
        ),
        pytest.param(
            nwp.when(a=1).then(10).otherwise(nwp.col("c").alias("c?")),
            pl.when(a=1).then(10).otherwise(pl.col("c").alias("c?")),
            "literal",
            id="When-Literal-Alias",
        ),
        pytest.param(
            (
                nwp.when(nwp.col("a").alias("a?"))
                .then(1)
                .when(nwp.col("b") == 1)
                .then(nwp.col("c"))
            ),
            (
                pl.when(pl.col("a").alias("a?"))
                .then(1)
                .when(pl.col("b") == 1)
                .then(pl.col("c"))
            ),
            "literal",
            id="When-Literal-BinaryExpr-Column",
        ),
        pytest.param(
            (
                nwp.when(nwp.col("foo") > 2, nwp.col("bar") < 3)
                .then(nwp.lit("Yes"))
                .otherwise(nwp.lit("No"))
                .alias("TARGET")
            ),
            (
                pl.when(pl.col("foo") > 2, pl.col("bar") < 3)
                .then(pl.lit("Yes"))
                .otherwise(pl.lit("No"))
                .alias("TARGET")
            ),
            "TARGET",
            id="When2-Literal-Literal-Alias",
        ),
        pytest.param(
            (nwp.col("ROOT").alias("ROOT-ALIAS").filter(nwp.col("c") <= 1).mean()),
            (pl.col("ROOT").alias("ROOT-ALIAS").filter(pl.col("c") <= 1).mean()),
            "ROOT-ALIAS",
            id="Filter",
        ),
        pytest.param(
            nwp.int_range(0, 10), pl.int_range(0, 10), "literal", id="IntRange-Literal"
        ),
        pytest.param(
            nwp.int_range(nwp.col("b").first(), 10),
            pl.int_range(pl.col("b").first(), 10),
            "b",
            id="IntRange-Column",
        ),
    ],
)
def test_meta_output_name(nw_expr: nwp.Expr, pl_expr: pl.Expr, expected: str) -> None:
    pl_result = pl_expr.meta.output_name()
    nw_result = nw_expr.meta.output_name()
    assert nw_result == expected
    assert nw_result == pl_result


def test_root_and_output_names() -> None:
    e = nwp.col("foo") * nwp.col("bar")
    assert e.meta.output_name() == "foo"
    assert e.meta.root_names() == ["foo", "bar"]

    e = nwp.col("foo").filter(bar=13)
    assert e.meta.output_name() == "foo"
    assert e.meta.root_names() == ["foo", "bar"]

    e = nwp.sum("foo").over("groups")
    assert e.meta.output_name() == "foo"
    assert e.meta.root_names() == ["foo", "groups"]

    e = nwp.sum("foo").is_between(nwp.len() - 10, nwp.col("bar"))
    assert e.meta.output_name() == "foo"
    assert e.meta.root_names() == ["foo", "bar"]

    e = nwp.len()
    assert e.meta.output_name() == "len"

    with pytest.raises(
        ComputeError,
        match=re.escape(
            "unable to find root column name for expr 'ncs.all()' when calling 'output_name'"
        ),
    ):
        nwp.all().name.suffix("_").meta.output_name()

    assert (
        nwp.all().name.suffix("_").meta.output_name(raise_if_undetermined=False) is None
    )


def test_meta_has_multiple_outputs() -> None:
    e = nwp.col(["a", "b"]).name.suffix("_foo")
    assert e.meta.has_multiple_outputs()


def test_is_column() -> None:
    e = nwp.col("foo")
    assert e.meta.is_column()

    e = nwp.col("foo").alias("bar")
    assert not e.meta.is_column()

    e = nwp.col("foo") * nwp.col("bar")
    assert not e.meta.is_column()


# TODO @dangotbanned: Uncomment the cases as they're added
@pytest.mark.parametrize(
    ("expr", "is_column_selection"),
    [
        # columns
        (nwp.col("foo"), True),
        (nwp.col("foo", "bar"), True),
        # column expressions
        (nwp.col("foo") + 100, False),
        (nwp.col("foo").__floordiv__(10), False),
        (nwp.col("foo") * nwp.col("bar"), False),
        # selectors / expressions
        (ncs.numeric() * 100, False),
        # (ncs.temporal() - ncs.time(), True),  # noqa: ERA001
        (ncs.numeric().exclude("value"), True),
        # ((ncs.temporal() - ncs.time()).exclude("dt"), True),  # noqa: ERA001
        # top-level selection funcs
        (nwp.nth(2), True),
        # (nwp.first(), True),  # noqa: ERA001
        # (nwp.last(), True),  # noqa: ERA001
    ],
)
def test_is_column_selection(expr: nwp.Expr, *, is_column_selection: bool) -> None:
    if is_column_selection:
        assert expr.meta.is_column_selection()
        assert expr.meta.is_column_selection(allow_aliasing=True)
        expr = (
            expr.name.suffix("!") if expr.meta.has_multiple_outputs() else expr.alias("!")
        )
        assert not expr.meta.is_column_selection()
        assert expr.meta.is_column_selection(allow_aliasing=True)
    else:
        assert not expr.meta.is_column_selection()


@pytest.mark.parametrize(
    "value",
    [
        None,
        1234,
        567.89,
        float("inf"),
        dt.date(2000, 1, 1),
        dt.datetime(1974, 1, 1, 12, 45, 1),
        dt.time(10, 30, 45),
        dt.timedelta(hours=-24),
        pytest.param(["x", "y", "z"], marks=XFAIL_LITERAL_LIST),
        series([None, None]),
        pytest.param([[10, 20], [30, 40]], marks=XFAIL_LITERAL_LIST),
        "this is the way",
    ],
)
def test_is_literal(value: Any) -> None:
    e = nwp.lit(value)
    assert e.meta.is_literal()

    e = nwp.lit(value).alias("foo")
    assert not e.meta.is_literal()

    e = nwp.lit(value).alias("foo")
    assert e.meta.is_literal(allow_aliasing=True)


def test_literal_output_name() -> None:
    e = nwp.lit(1)
    data = 1, 2, 3
    assert e.meta.output_name() == "literal"

    e = nwp.lit(series(data).alias("abc"))
    assert e.meta.output_name() == "abc"

    e = nwp.lit(series(data))
    assert e.meta.output_name() == ""


# NOTE: Very low-priority
@pytest.mark.xfail(
    reason="TODO: `Expr.struct.field` influences `meta.output_name`.",
    raises=AssertionError,
)
def test_struct_field_output_name_24003() -> None:
    assert nwp.col("ball").struct.field("radius").meta.output_name() == "radius"


def test_selector_by_name_single() -> None:
    assert ncs.by_name("foo").meta.output_name() == "foo"


def test_selector_by_name_multiple() -> None:
    with pytest.raises(
        ComputeError,
        match=re.escape(
            "unable to find root column name for expr 'ncs.by_name('foo', 'bar', require_all=True)' when calling 'output_name'"
        ),
    ):
        ncs.by_name(["foo", "bar"]).meta.output_name()
