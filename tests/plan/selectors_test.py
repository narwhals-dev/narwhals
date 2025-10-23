"""Tests adapted from [upstream].

[upstream]: https://github.com/pola-rs/polars/blob/84d66e960e3d462811f0575e0a6e4e78e34c618c/py-polars/tests/unit/test_selectors.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals import _plan as nwp
from narwhals._plan import Selector, expressions as ir, selectors as ncs
from narwhals._plan._expansion import prepare_projection
from narwhals._plan._parse import parse_into_seq_of_expr_ir
from narwhals.exceptions import ColumnNotFoundError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from narwhals._plan.typing import IntoExpr, Seq
    from narwhals.typing import IntoDType


XFAIL_NESTED_INNER_SELECTOR = pytest.mark.xfail(
    reason="Bug causing the inner selector to always be falsy?", raises=AssertionError
)
XFAIL_REQUIRE_ALL = pytest.mark.xfail(reason="Strict selectors not yet implemented")
XFAIL_BY_INDEX_MATCHES = pytest.mark.xfail(
    reason="`cs.by_index()` matching is only partially implemented",
    raises=NotImplementedError,
)

XFAIL_REORDERING = pytest.mark.xfail(
    reason="`cs.by_{index,name}()` reordering is only partially implemented"
)


def _schema(mapping: Mapping[str, IntoDType]) -> nw.Schema:
    # NOTE: Runtime isn't as strict as the annotation which requires instantiated `DType`
    return nw.Schema(mapping)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def schema_nested_1() -> nw.Schema:
    return _schema(
        {
            "a": nw.Int32(),
            "b": nw.List(nw.Int32()),
            "c": nw.List(nw.UInt32),
            "d": nw.Array(nw.Int32, 3),
            "e": nw.List(nw.String),
            "f": nw.Struct({"x": nw.Int32}),
        }
    )


@pytest.fixture(scope="module")
def schema_nested_2() -> nw.Schema:
    return _schema(
        {
            "a": nw.Int32(),
            "b": nw.Array(nw.Int32, 4),
            "c": nw.Array(nw.UInt32, 4),
            "d": nw.Array(nw.Int32, 3),
            "e": nw.List(nw.Int32),
            "f": nw.Array(nw.String, 4),
            "g": nw.Struct({"x": nw.Int32}),
        }
    )


@pytest.fixture(scope="module")
def schema_non_nested() -> nw.Schema:
    return _schema(
        {
            "abc": nw.UInt16(),
            "bbb": nw.UInt32(),
            "cde": nw.Float64(),
            "def": nw.Float32(),
            "eee": nw.Boolean(),
            "fgg": nw.Boolean(),
            "ghi": nw.Time(),
            "JJK": nw.Date(),
            "Lmn": nw.Duration(),
            "opp": nw.Datetime("ms"),
            "qqR": nw.String(),
        }
    )


class Frame:
    def __init__(self, schema: nw.Schema) -> None:
        self.schema = schema
        self.columns = tuple(schema.names())

    @property
    def width(self) -> int:
        return len(self.columns)

    def project_named_irs(self, *exprs: IntoExpr) -> Seq[ir.NamedIR]:
        expr_irs = parse_into_seq_of_expr_ir(*exprs)
        named_irs, _ = prepare_projection(expr_irs, schema=self.schema)
        return named_irs

    def project_names(self, *exprs: IntoExpr) -> Seq[str]:
        named_irs = self.project_named_irs(*exprs)
        return tuple(e.name for e in named_irs)

    def assert_selects(self, selector: Selector, *column_names: str) -> None:
        projected = self.project_names(selector)
        expected = column_names
        assert projected == expected


def test_selector_all(schema_non_nested: nw.Schema) -> None:
    df = Frame(schema_non_nested)

    df.assert_selects(ncs.all(), *df.columns)
    df.assert_selects(~ncs.all())
    df.assert_selects(~(~ncs.all()), *df.columns)

    # TODO @dangotbanned: Fix typing, this returns a `Selector` at runtime
    selector_and_col = ncs.all() & nwp.col("abc")
    df.assert_selects(selector_and_col, "abc")  # type: ignore[arg-type]


def test_selector_by_dtype(schema_non_nested: nw.Schema) -> None:
    df = Frame(schema_non_nested)

    selector = ncs.boolean() | ncs.by_dtype(nw.UInt16)
    df.assert_selects(selector, "abc", "eee", "fgg")

    selector = ~ncs.by_dtype(
        nw.Int8,
        nw.Int16,
        nw.Int32,
        nw.Int64,
        nw.Int128,
        nw.UInt8,
        nw.UInt16,
        nw.UInt32,
        nw.UInt64,
        nw.UInt128,
        nw.Date,
        nw.Datetime,
        nw.Duration,
        nw.Time,
    )
    df.assert_selects(selector, "cde", "def", "eee", "fgg", "qqR")

    selector = ncs.by_dtype(nw.Datetime("ns"), nw.Float32, nw.UInt32, nw.Date)
    df.assert_selects(selector, "bbb", "def", "JJK")


def test_selector_by_dtype_timezone_decimal() -> None:
    schema = nw.Schema(
        {
            "idx": nw.Decimal(),
            "dt1": nw.Datetime("ms"),
            "dt2": nw.Datetime(time_zone="Asia/Tokyo"),
        }
    )
    df = Frame(schema)
    df.assert_selects(ncs.by_dtype(nw.Decimal), "idx")
    df.assert_selects(ncs.by_dtype(nw.Datetime(time_zone="Asia/Tokyo")), "dt2")
    df.assert_selects(ncs.by_dtype(nw.Datetime("ms", None)), "dt1")
    df.assert_selects(ncs.by_dtype(nw.Datetime), "dt1", "dt2")


@pytest.mark.xfail(
    reason="Bug: Forgot to handle this during construction", raises=StopIteration
)
def test_selector_by_dtype_empty(
    schema_non_nested: nw.Schema,
) -> None:  # pragma: no cover
    df = Frame(schema_non_nested)
    # empty selection selects nothing
    df.assert_selects(ncs.by_dtype())
    df.assert_selects(ncs.by_dtype([]))


@pytest.mark.xfail(reason="Bug: Forgot to handle this during construction")
def test_selector_by_dtype_invalid_input() -> None:
    with pytest.raises(TypeError):
        ncs.by_dtype(999)  # type: ignore[arg-type]


@XFAIL_BY_INDEX_MATCHES
def test_selector_by_index(schema_non_nested: nw.Schema) -> None:  # pragma: no cover
    df = Frame(schema_non_nested)

    # # one or more positive indices
    df.assert_selects(ncs.by_index(0), "abc")
    df.assert_selects(nwp.nth(0, 1, 2), "abc", "bbb", "cde")  # type: ignore[arg-type]
    df.assert_selects(ncs.by_index(0, 1, 2), "abc", "bbb", "cde")

    # one or more negative indices
    df.assert_selects(ncs.by_index(-1), "qqR")

    # range objects
    df.assert_selects(ncs.by_index(range(3)), "abc", "bbb", "cde")

    # exclude by index
    df.assert_selects(
        ~ncs.by_index(range(0, df.width, 2)), "bbb", "def", "fgg", "JJK", "opp"
    )


@pytest.mark.xfail(reason="Bug: Forgot to handle this during construction")
def test_selector_by_index_invalid_input() -> None:  # pragma: no cover
    with pytest.raises(TypeError):
        ncs.by_index("one")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        ncs.by_index(["two", "three"])  # type: ignore[list-item]


@XFAIL_REQUIRE_ALL
def test_selector_by_index_not_found(schema_non_nested: nw.Schema) -> None:
    df = Frame(schema_non_nested)

    with pytest.raises(ColumnNotFoundError):
        df.project_named_irs(ncs.by_index(999))


@XFAIL_REORDERING
def test_selector_by_index_reordering(
    schema_non_nested: nw.Schema,
) -> None:  # pragma: no cover
    df = Frame(schema_non_nested)

    df.assert_selects(ncs.by_index(-3, -2, -1), "Lmn", "opp", "qqR")
    df.assert_selects(ncs.by_index(range(-3, 0)), "abc", "Lmn", "opp", "qqR")


@XFAIL_NESTED_INNER_SELECTOR
def test_list_selector(schema_nested_1: nw.Schema) -> None:  # pragma: no cover
    df = Frame(schema_nested_1)
    df.assert_selects(ncs.list(), "b", "c", "e")

    # NOTE: bug here
    df.assert_selects(ncs.list(inner=ncs.numeric()), "b", "c")
    df.assert_selects(ncs.list(inner=ncs.string()), "e")

    # NOTE: Not implemented
    with pytest.raises(
        TypeError, match=r"expected datatype based expression got.+by_name\("
    ):
        df.project_named_irs(ncs.list(inner=ncs.by_name("???")))


@XFAIL_NESTED_INNER_SELECTOR
def test_array_selector(schema_nested_2: nw.Schema) -> None:  # pragma: no cover
    df = Frame(schema_nested_2)
    df.assert_selects(ncs.array(), "b", "c", "d", "f")
    df.assert_selects(ncs.array(size=4), "b", "c", "f")

    # NOTE: bug here
    df.assert_selects(ncs.array(inner=ncs.numeric()), "b", "c", "d")
    df.assert_selects(ncs.array(inner=ncs.string()), "f")

    # NOTE: Not implemented
    with pytest.raises(
        TypeError, match=r"expected datatype based expression got.+by_name\("
    ):
        df.project_named_irs(ncs.array(inner=ncs.by_name("???")))
