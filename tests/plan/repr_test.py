from __future__ import annotations

import datetime as dt
import decimal
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
import narwhals._plan as nwp
import narwhals.stable.v1 as nw_v1
from tests.plan.utils import Series, assert_expr_ir_equal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import LiteralString


def test_expr_levels() -> None:
    nwp.col("a").meta.as_selector()
    expr = nwp.col("a")
    selector = expr.meta.as_selector()

    expr_repr_html = expr._repr_html_()
    expr_ir_repr_html = expr._ir._repr_html_()
    selector_repr_html = selector._repr_html_()
    selector_ir_repr_html = selector._ir._repr_html_()
    expr_repr = expr.__repr__()
    expr_ir_repr = expr._ir.__repr__()
    selector_repr = selector.__repr__()
    selector_ir_repr = selector._ir.__repr__()

    # In a notebook, both `Expr` and `ExprIR` are displayed the same
    assert expr_repr_html == expr_ir_repr_html
    # The actual repr (for debugging) has more information
    assert expr_repr != expr_repr_html
    # Currently, all extra information is *before* the part which matches
    assert expr_repr.endswith(expr_repr_html)
    # But these guys should not deviate
    assert expr_ir_repr == expr_ir_repr_html
    # The same invariants should hold for `Selector` and `SelectorIR`
    assert selector_repr_html == selector_ir_repr_html
    assert selector_repr != selector_repr_html
    assert selector_repr.endswith(selector_repr_html)
    assert selector_ir_repr == selector_ir_repr_html
    # But they must still be visually different from `Expr` and `ExprIR`
    assert selector_repr_html != expr_repr_html
    assert selector_repr != expr_repr


def test_expr_ir_nodes() -> None:
    a = nwp.col("a")
    b = a.first()
    c = b.over("c")

    a_nodes = a._ir.__expr_ir_nodes__
    b_nodes = b._ir.__expr_ir_nodes__
    c_nodes = c._ir.__expr_ir_nodes__

    a_nodes_repr = repr(a_nodes)
    b_nodes_repr = repr(b_nodes)
    c_nodes_repr = repr(c_nodes)
    a_nodes_repr_html = a_nodes._repr_html_()
    b_nodes_repr_html = b_nodes._repr_html_()
    c_nodes_repr_html = c_nodes._repr_html_()

    assert a_nodes_repr == a_nodes_repr_html
    assert b_nodes_repr != b_nodes_repr_html
    assert "    " in b_nodes_repr
    assert "    " not in b_nodes_repr_html
    assert "&nbsp;" not in b_nodes_repr
    assert "&nbsp;" in b_nodes_repr_html

    assert c_nodes_repr != c_nodes_repr_html
    assert "\n" in c_nodes_repr
    assert "\n" not in c_nodes_repr_html
    assert "<br>" not in c_nodes_repr
    assert "<br>" in c_nodes_repr_html


@pytest.mark.parametrize(
    ("exprs", "expected"),
    [
        (
            [
                nwp.lit(1),
                nwp.lit(1, nw.UInt8),
                nwp.lit(1, nw.Int32),
                nwp.lit(None, nw.Int64),
            ],
            "[lit(1), lit[u8](1), lit[i32](1), lit[i64](None)]",
        ),
        (nwp.int_range(nwp.len()), "int_range([lit(0), len()])"),
        (nwp.int_range(0, 10), "int_range([lit(0), lit(10)])"),
        (
            [
                nwp.lit(1.479).alias("renamed"),
                nwp.lit(14.2, nw.Float32),
                nwp.lit(None, nw.Float64),
            ],
            "[lit(1.479).alias('renamed'), lit[f32](14.2), lit[f64](None)]",
        ),
        (
            (
                nwp.col("one"),
                nwp.lit(["two"]),
                nwp.lit(None, nw.List(nw.String)),
                nwp.lit([], nw.List(nw.String)),
                nwp.lit(["a", "b", "c", "d", "e"]),
                nwp.lit(["a", None, "c"]),
            ),
            "[col('one'), "
            "lit[list](['two']), "
            "lit[list[str]](None), "
            "lit[list[str]]([]), "
            "lit[list[str]]([...]), "
            "lit[list](['a', None, 'c'])]",
        ),
        (
            (
                nwp.lit(dt.date(2000, 1, 1)),
                nwp.lit(None, nw.Date),
                nwp.lit(dt.time(9, 30, 2, 9)),
                nwp.lit(None, nw.Time),
            ),
            "[lit[date]('2000-01-01'), "
            "lit[date](None), "
            "lit[time]('09:30:02.000009'), "
            "lit[time](None)]",
        ),
        (
            (
                nwp.lit(dt.datetime(2032, 8, 29, 14, 40, 26, 10)),
                nwp.lit(dt.datetime(2010, 8, 29), nw.Datetime("ns")),
                nwp.lit(dt.datetime(1986, 1, 1, 1, 1, 1), nw.Datetime(time_zone="UTC")),
            ),
            "[lit[datetime]('2032-08-29T14:40:26.000010'), "
            "lit[datetime[ns]]('2010-08-29T00:00:00'), "
            "lit[datetime[us, UTC]]('1986-01-01T01:01:01')]",
        ),
        (
            [
                nwp.lit(dt.timedelta(12)),
                nwp.lit(dt.timedelta(5, 1, 5)),
                nwp.lit(dt.timedelta(99), nw.Duration("ms")),
                nwp.lit(dt.timedelta()),
                nwp.lit(dt.timedelta(seconds=123)),
                nwp.lit(dt.timedelta(seconds=456, microseconds=789)),
                nwp.lit(None, nw.Duration("s")),
                nwp.lit(None, nw.Duration),
            ],
            "[lit[duration]('12d'), "
            "lit[duration]('5d 1s 5us'), "
            "lit[duration[ms]]('99d'), "
            "lit[duration]('0'), "
            "lit[duration]('123s'), "
            "lit[duration]('456s 789us'), "
            "lit[duration[s]](None), "
            "lit[duration](None)]",
        ),
        (
            [
                nwp.lit(decimal.Decimal("0.37392")),
                nwp.lit(decimal.Decimal("0.37392"), nw.Decimal(5)),
                nwp.lit(decimal.Decimal("0.37392"), nw.Decimal(5, 1)),
                nwp.lit(None, nw.Decimal),
                nwp.lit(None, nw.Decimal(4, 2)),
            ],
            "[lit[decimal]('0.37392'), "
            "lit[decimal[5,0]]('0.37392'), "
            "lit[decimal[5,1]]('0.37392'), "
            "lit[decimal](None), "
            "lit[decimal[4,2]](None)]",
        ),
        (
            [
                nwp.lit("abcdef"),
                nwp.lit(b"abcdef"),
                nwp.lit(None, nw.String),
                nwp.lit(None, nw.Binary),
            ],
            "[lit('abcdef'), lit(b'abcdef'), lit[str](None), lit[binary](None)]",
        ),
        (
            [nwp.lit(True), nwp.lit(False), nwp.lit(None, nw.Boolean)],
            "[lit(True), lit(False), lit[bool](None)]",
        ),
        (
            [nwp.lit("a", nw.Categorical), nwp.lit(None, nw.Categorical)],
            "[lit[cat]('a'), lit[cat](None)]",
        ),
        (
            [
                nwp.lit("a", nw.Enum(["a", "b"])),
                nwp.lit("a", nw_v1.Enum()),
                nwp.lit(None, nw.Enum(["a", "b"])),
                nwp.lit(None, nw_v1.Enum()),
            ],
            "[lit[enum]('a'), lit[enum]('a'), lit[enum](None), lit[enum](None)]",
        ),
        (
            [
                nwp.lit(
                    {"hello": None, "there": 99},
                    nw.Struct({"hello": nw.Array(nw.Int64, 1), "there": nw.UInt128}),
                ),
                nwp.lit({"a": 1, "b": 5, "c": 10}),
                nwp.lit(None, nw.Struct({"a": nw.Boolean})),
                nwp.lit({}, nw.Struct([])),
            ],
            "[lit[struct[2]]({'hello': None, 'there': 99}), "
            "lit[struct[3]]({'a': 1, 'b': 5, 'c': 10}), "
            "lit[struct[1]](None), "
            "lit[struct[0]]({})]",
        ),
        (
            [
                nwp.lit([1, 2, 3], nw.Array(nw.Int64, 3)),
                nwp.lit([1, None], nw.Array(nw.Int32, 2)),
                nwp.lit(["hi"], nw.Array(nw.String, 1)),
                nwp.lit([None, None, None, None], nw.Array(nw.Float32, 4)),
                nwp.lit((1, 2, 3, 4, 5), nw.Array(nw.UInt8, 5)),
            ],
            "[lit[array[i64, 3]]([1, 2, 3]), "
            "lit[array[i32, 2]]([1, None]), "
            "lit[array[str, 1]](['hi']), "
            "lit[array[f32, 4]]([None, None, None, None]), "
            "lit[array[u8, 5]]([...])]",
        ),
    ],
)
def test_lit(exprs: nwp.Expr | Sequence[nwp.Expr], expected: LiteralString) -> None:
    # NOTE: Checking both how `lit` looks like in isolation, and when appearing inside/alongside other expressions
    # The shape of the test code is intended to make the visual comparison in the `parametrize` cases easier to read
    if isinstance(exprs, nwp.Expr):
        exprs = (exprs,)
    if len(exprs) == 1:
        assert_expr_ir_equal(exprs[0], expected)
    else:
        actual = "[" + (", ".join(repr(e._ir) for e in exprs)) + "]"
        assert actual == expected


def test_lit_series(series: Series) -> None:
    # NOTE: Don't make this parametric, the idea is to see the full string
    if series.is_polars():
        expected = "lit(Series[pl.Series])"
    elif series.is_pyarrow():
        expected = "lit(Series[pa.ChunkedArray])"
    else:
        raise NotImplementedError(series.identifier)
    assert_expr_ir_equal(nwp.lit(series([True, False, True])), expected)


def test_lit_object() -> None:
    class What:
        def __repr__(self) -> str:
            return "12345"

    obj = What()
    expr = nwp.lit(obj, nw.Object)
    assert_expr_ir_equal(expr, "lit[object](12345)")
