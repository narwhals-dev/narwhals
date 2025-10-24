"""Tests adapted from [upstream].

[upstream]: https://github.com/pola-rs/polars/blob/84d66e960e3d462811f0575e0a6e4e78e34c618c/py-polars/tests/unit/test_selectors.py
"""

from __future__ import annotations

from datetime import timezone
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from narwhals import _plan as nwp
from narwhals._plan import Expr, Selector, expressions as ir, selectors as ncs
from narwhals._plan._expansion import prepare_projection
from narwhals._plan._parse import parse_into_seq_of_expr_ir
from narwhals.exceptions import ColumnNotFoundError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from narwhals._plan.typing import IntoExpr, Seq
    from narwhals.typing import IntoDType, IntoSchema


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

    @staticmethod
    def from_mapping(mapping: IntoSchema) -> Frame:
        return Frame(nw.Schema(mapping))

    @staticmethod
    def from_names(*column_names: str) -> Frame:
        """Construct with all `nw.Int64()`."""
        return Frame(nw.Schema((name, nw.Int64()) for name in column_names))

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

    def assert_selects(self, selector: Selector | Expr, *column_names: str) -> None:
        projected = self.project_names(selector)
        expected = column_names
        assert projected == expected


@pytest.fixture(scope="module")
def df_datetime() -> Frame:
    return Frame.from_mapping(
        {
            "d1": nw.Datetime("ns", "Asia/Tokyo"),
            "d2": nw.Datetime("ns", "UTC"),
            "d3": nw.Datetime("us", "UTC"),
            "d4": nw.Datetime("us"),
            "d5": nw.Datetime("ms"),
        }
    )


def test_selector_all(schema_non_nested: nw.Schema) -> None:
    df = Frame(schema_non_nested)

    df.assert_selects(ncs.all(), *df.columns)
    df.assert_selects(~ncs.all())
    df.assert_selects(~(~ncs.all()), *df.columns)

    selector_and_col = ncs.all() & nwp.col("abc")
    df.assert_selects(selector_and_col, "abc")


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
    df = Frame.from_mapping(
        {
            "idx": nw.Decimal(),
            "dt1": nw.Datetime("ms"),
            "dt2": nw.Datetime(time_zone="Asia/Tokyo"),
        }
    )
    df.assert_selects(ncs.by_dtype(nw.Decimal), "idx")
    df.assert_selects(ncs.by_dtype(nw.Datetime(time_zone="Asia/Tokyo")), "dt2")
    df.assert_selects(ncs.by_dtype(nw.Datetime("ms", None)), "dt1")
    df.assert_selects(ncs.by_dtype(nw.Datetime), "dt1", "dt2")


def test_selector_by_dtype_empty(schema_non_nested: nw.Schema) -> None:
    df = Frame(schema_non_nested)
    # empty selection selects nothing
    df.assert_selects(ncs.by_dtype())
    df.assert_selects(ncs.by_dtype([]))


def test_selector_by_dtype_invalid_input() -> None:
    with pytest.raises(TypeError):
        ncs.by_dtype(999)  # type: ignore[arg-type]


@XFAIL_BY_INDEX_MATCHES
def test_selector_by_index(schema_non_nested: nw.Schema) -> None:  # pragma: no cover
    df = Frame(schema_non_nested)

    WIDTH_COVERAGE = df.width  # noqa: N806

    # # one or more positive indices
    df.assert_selects(ncs.by_index(0), "abc")
    df.assert_selects(nwp.nth(0, 1, 2), "abc", "bbb", "cde")
    df.assert_selects(ncs.by_index(0, 1, 2), "abc", "bbb", "cde")

    # one or more negative indices
    df.assert_selects(ncs.by_index(-1), "qqR")

    # range objects
    df.assert_selects(ncs.by_index(range(3)), "abc", "bbb", "cde")

    # exclude by index
    df.assert_selects(
        ~ncs.by_index(range(0, WIDTH_COVERAGE, 2)), "bbb", "def", "fgg", "JJK", "opp"
    )


def test_selector_by_index_invalid_input() -> None:
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


def test_selector_by_name(schema_non_nested: nw.Schema) -> None:
    df = Frame(schema_non_nested)

    df.assert_selects(ncs.by_name("abc", "cde"), "abc", "cde")

    selector = ~ncs.by_name("abc", "cde", "ghi", "Lmn", "opp", "eee")
    df.assert_selects(selector, "bbb", "def", "fgg", "JJK", "qqR")
    df.assert_selects(ncs.by_name())
    df.assert_selects(ncs.by_name([]))

    df.assert_selects(ncs.by_name("???", "fgg", "!!!", require_all=False), "fgg")

    df.assert_selects(ncs.by_name("missing", require_all=False))
    df.assert_selects(ncs.by_name("???", require_all=False))

    # check "by_name & col"
    df.assert_selects(ncs.by_name("abc", "cde") & nwp.col("ghi"))
    df.assert_selects(ncs.by_name("abc", "cde") & nwp.col("cde"), "cde")
    df.assert_selects(ncs.by_name("cde") & ncs.by_name("cde", "abc"), "cde")

    # check "by_name & by_name"
    selector = ncs.by_name("abc", "cde", "def", "eee") & ncs.by_name("cde", "eee", "fgg")
    df.assert_selects(selector, "cde", "eee")


def test_selector_by_name_or_col(schema_non_nested: nw.Schema) -> None:
    df = Frame(schema_non_nested)
    df.assert_selects(ncs.by_name("abc") | nwp.col("cde"), "abc", "cde")


@XFAIL_REQUIRE_ALL
def test_selector_by_name_not_found(
    schema_non_nested: nw.Schema,
) -> None:  # pragma: no cover
    df = Frame(schema_non_nested)

    with pytest.raises(ColumnNotFoundError):
        df.project_named_irs(ncs.by_name("xxx", "fgg", "!!!"))

    with pytest.raises(ColumnNotFoundError):
        df.project_named_irs(ncs.by_name("stroopwafel"))


def test_selector_by_name_invalid_input() -> None:
    with pytest.raises(TypeError):
        ncs.by_name(999)  # type: ignore[arg-type]


def test_selector_datetime(schema_non_nested: nw.Schema) -> None:
    df = Frame(schema_non_nested)
    df.assert_selects(ncs.datetime(), "opp")
    df.assert_selects(ncs.datetime("ns"))
    all_columns = list(df.columns)
    all_columns.remove("opp")
    df.assert_selects(~ncs.datetime(), *all_columns)


# TODO @dangotbanned: Use `parametrize`, the test is waaaaaay too long
# https://github.com/pola-rs/polars/blob/84d66e960e3d462811f0575e0a6e4e78e34c618c/py-polars/tests/unit/test_selectors.py#L283
def test_selector_datetime_exhaustive(df_datetime: Frame) -> None:
    df = df_datetime

    df.assert_selects(ncs.datetime(), "d1", "d2", "d3", "d4", "d5")
    df.assert_selects(~ncs.datetime())
    df.assert_selects(ncs.datetime(["ms", "ns"]), "d1", "d2", "d5")
    df.assert_selects(ncs.datetime(["ms", "ns"], time_zone="*"), "d1", "d2")
    df.assert_selects(~ncs.datetime(["ms", "ns"]), "d3", "d4")
    df.assert_selects(~ncs.datetime(["ms", "ns"], time_zone="*"), "d3", "d4", "d5")
    df.assert_selects(
        ncs.datetime(time_zone=["UTC", "Asia/Tokyo", "Europe/London"]), "d1", "d2", "d3"
    )
    df.assert_selects(ncs.datetime(time_zone="*"), "d1", "d2", "d3")
    df.assert_selects(ncs.datetime("ns", time_zone="*"), "d1", "d2")
    df.assert_selects(ncs.datetime(time_zone="UTC"), "d2", "d3")
    df.assert_selects(ncs.datetime("us", time_zone="UTC"), "d3")
    df.assert_selects(ncs.datetime(time_zone="Asia/Tokyo"), "d1")
    df.assert_selects(ncs.datetime("us", time_zone="Asia/Tokyo"))
    df.assert_selects(ncs.datetime(time_zone=None), "d4", "d5")
    df.assert_selects(ncs.datetime("ns", time_zone=None))
    df.assert_selects(~ncs.datetime(time_zone="*"), "d4", "d5")
    df.assert_selects(~ncs.datetime("ns", time_zone="*"), "d3", "d4", "d5")
    df.assert_selects(~ncs.datetime(time_zone="UTC"), "d1", "d4", "d5")
    df.assert_selects(~ncs.datetime("us", time_zone="UTC"), "d1", "d2", "d4", "d5")
    df.assert_selects(~ncs.datetime(time_zone="Asia/Tokyo"), "d2", "d3", "d4", "d5")
    df.assert_selects(
        ~ncs.datetime("us", time_zone="Asia/Tokyo"), "d1", "d2", "d3", "d4", "d5"
    )
    df.assert_selects(~ncs.datetime(time_zone=None), "d1", "d2", "d3")
    df.assert_selects(~ncs.datetime("ns", time_zone=None), "d1", "d2", "d3", "d4", "d5")
    df.assert_selects(ncs.datetime("ns"), "d1", "d2")
    df.assert_selects(ncs.datetime("us"), "d3", "d4")
    df.assert_selects(ncs.datetime("ms"), "d5")


# NOTE: The test is *technically* passing, but the `TypeError` is being raised by `set(time_unit)`
# `TypeError: 'int' object is not iterable`
def test_selector_datetime_invalid_input() -> None:
    with pytest.raises(TypeError):
        ncs.datetime(999)  # type: ignore[arg-type]


def test_selector_duration(schema_non_nested: nw.Schema) -> None:
    df = Frame(schema_non_nested)

    df.assert_selects(ncs.duration("ms"))
    df.assert_selects(ncs.duration(["ms", "ns"]))
    df.assert_selects(ncs.duration(), "Lmn")

    df = Frame.from_mapping(
        {"d1": nw.Duration("ns"), "d2": nw.Duration("us"), "d3": nw.Duration("ms")}
    )
    df.assert_selects(ncs.duration("us"), "d2")
    df.assert_selects(ncs.duration(["ms", "ns"]), "d1", "d3")
    df.assert_selects(ncs.duration(), "d1", "d2", "d3")


# TODO @dangotbanned: Support passing in `re.Pattern` for more control?
def test_selector_matches(schema_non_nested: nw.Schema) -> None:
    # NOTE: Slightly modified from `polars`, because python's `re` raises on `^(?i)[E-N]{3}$`
    # re.PatternError: global flags not at the start of the expression at position 1
    df = Frame(schema_non_nested)

    df.assert_selects(ncs.matches(r"(?i)[E-N]{3}"), "eee", "fgg", "ghi", "JJK", "Lmn")
    df.assert_selects(
        ~ncs.matches(r"(?i)[E-N]{3}"), "abc", "bbb", "cde", "def", "opp", "qqR"
    )


def test_selector_categorical(schema_non_nested: nw.Schema) -> None:
    df = Frame(schema_non_nested)
    df.assert_selects(ncs.categorical())

    df = Frame.from_mapping({"a": nw.String(), "b": nw.Binary(), "c": nw.Categorical()})
    df.assert_selects(ncs.categorical(), "c")
    df.assert_selects(~ncs.categorical(), "a", "b")


def test_selector_numeric(schema_non_nested: nw.Schema) -> None:
    df = Frame(schema_non_nested)
    df.assert_selects(ncs.numeric(), "abc", "bbb", "cde", "def")
    df.assert_selects(ncs.numeric() - ncs.by_dtype(nw.UInt16), "bbb", "cde", "def")
    df.assert_selects(~ncs.numeric(), "eee", "fgg", "ghi", "JJK", "Lmn", "opp", "qqR")


def test_selector_expansion() -> None:
    # https://github.com/pola-rs/polars/blob/84d66e960e3d462811f0575e0a6e4e78e34c618c/py-polars/tests/unit/test_selectors.py#L619
    df = Frame.from_names(*list("abcde"))

    s1 = nwp.all().meta.as_selector()
    s2 = nwp.col(["a", "b"]).meta.as_selector()
    s = s1 - s2
    df.assert_selects(s, "c", "d", "e")

    s1 = ncs.matches("^a|b$").meta.as_selector()
    s = s1 | nwp.col(["d", "e"]).meta.as_selector()
    df.assert_selects(s, "a", "b", "d", "e")

    s = s - nwp.col("d").meta.as_selector()
    df.assert_selects(s, "a", "b", "e")

    # add a duplicate, this tests if they are pruned
    s = s | nwp.col("a").meta.as_selector()
    df.assert_selects(s, "a", "b", "e")

    s1e = nwp.col(["a", "b", "c"])
    s2e = nwp.col(["b", "c", "d"])

    s = s1e.meta.as_selector()
    s = s & s2e.meta.as_selector()
    df.assert_selects(s, "b", "c")


def test_selector_sets(schema_non_nested: nw.Schema) -> None:
    df = Frame(schema_non_nested)

    # NOTE: `cs.temporal` is used a lot in this tests, but `narwhals` doesn't have it
    temporal = ncs.datetime() | ncs.duration() | ncs.by_dtype(nw.Date, nw.Time)

    # or
    selector = temporal | ncs.string() | ncs.matches(r"^e")
    df.assert_selects(selector, "eee", "ghi", "JJK", "Lmn", "opp", "qqR")

    # and
    selector = temporal & ncs.matches(r"opp|JJK")
    df.assert_selects(selector, "JJK", "opp")

    # SET A - SET B
    selector = temporal - ncs.matches(r"opp|JJK")
    df.assert_selects(selector, "ghi", "Lmn")
    # NOTE: `cs.exclude` was used, but `narwhals` doesn't have it
    # Would allow: `str | Expr | DType | type[DType] | Selector | Collection[str | Expr | DType | type[DType] | Selector]`
    selector = ncs.all() - (~temporal | ncs.matches(r"opp|JJK"))
    df.assert_selects(selector, "ghi", "Lmn")

    sub_expr = ncs.matches("[yz]$") - nwp.col("colx")
    assert not isinstance(sub_expr, Selector), (
        "('Selector' - 'Expr') shouldn't behave as set"
    )
    assert isinstance(sub_expr, Expr)

    with pytest.raises(TypeError, match=r"unsupported .* \('Expr' - 'Selector'\)"):
        nwp.col("colx") - ncs.matches("[yz]$")

    # complement
    selector = ~ncs.by_dtype([nw.Duration, nw.Time])
    df.assert_selects(
        selector, "abc", "bbb", "cde", "def", "eee", "fgg", "JJK", "opp", "qqR"
    )

    # exclusive or
    expected = "abc", "bbb", "eee", "fgg", "ghi"
    df.assert_selects(ncs.matches("e|g") ^ ncs.numeric(), *expected)
    df.assert_selects(ncs.matches(r"b|g") ^ nwp.col("eee"), *expected)


@pytest.mark.parametrize(
    "selector",
    [
        (ncs.string() | ncs.numeric()),
        (ncs.numeric() | ncs.string()),
        ~(~ncs.numeric() & ~ncs.string()),
        ~(~ncs.string() & ~ncs.numeric()),
        (ncs.by_dtype(nw.Int16) ^ ncs.matches(r"b|e|q")) - ncs.matches("^e"),
    ],
)
def test_selector_result_order(schema_non_nested: nw.Schema, selector: Selector) -> None:
    df = Frame(schema_non_nested)
    df.assert_selects(selector, "abc", "bbb", "cde", "def", "qqR")


@XFAIL_NESTED_INNER_SELECTOR
def test_selector_list(schema_nested_1: nw.Schema) -> None:  # pragma: no cover
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
def test_selector_array(schema_nested_2: nw.Schema) -> None:  # pragma: no cover
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


def test_selector_enum() -> None:
    df = Frame.from_mapping(
        {
            "a": nw.Int32(),
            "b": nw.UInt32(),
            "c": nw_v1.Enum(),
            "d": nw.Categorical(),
            "e": nw.String(),
            "f": nw.Enum(["a", "b"]),
        }
    )
    df.assert_selects(ncs.enum(), "c", "f")
    df.assert_selects(~ncs.enum(), "a", "b", "d", "e")


def test_selector_struct() -> None:
    df = Frame.from_mapping(
        {
            "a": nw.Int32(),
            "b": nw.Array(nw.Int32, shape=(4,)),
            "c": nw.Struct({}),
            "d": nw.Array(nw.UInt32, shape=(4,)),
            "e": nw.Struct({"x": nw.Int32, "y": nw.String}),
            "f": nw.List(nw.Int32),
            "g": nw.Array(nw.String, shape=(4,)),
            "h": nw.Struct({"x": nw.Int32}),
        }
    )
    df.assert_selects(ncs.struct(), "c", "e", "h")
    df.assert_selects(~ncs.struct(), "a", "b", "d", "f", "g")


def test_selector_matches_22816() -> None:
    df = Frame.from_names("ham", "hamburger", "foo", "bar")
    df.assert_selects(ncs.matches(r"^ham.*$"), "ham", "hamburger")
    df.assert_selects(ncs.matches(r".*burger"), "hamburger")


@XFAIL_REORDERING
def test_selector_by_name_order_19384() -> None:  # pragma: no cover
    df = Frame.from_names("a", "b")
    df.assert_selects(ncs.by_name("b", "a"), "b", "a")
    df.assert_selects(ncs.by_name("b", "a", require_all=False), "b", "a")


def test_selector_datetime_23767() -> None:
    df = Frame.from_mapping(
        {"a": nw.Datetime(), "b": nw.Datetime(time_zone=timezone.utc)}
    )
    df.assert_selects(ncs.datetime("us", time_zone=None), "a")
    df.assert_selects(ncs.datetime("us", time_zone=["UTC"]), "b")
    df.assert_selects(ncs.datetime("us", time_zone=[None, "UTC"]), "a", "b")
