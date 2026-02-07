"""Tests adapted from [upstream].

[upstream]: https://github.com/pola-rs/polars/blob/84d66e960e3d462811f0575e0a6e4e78e34c618c/py-polars/tests/unit/test_selectors.py
"""

from __future__ import annotations

import operator
import re
from datetime import timezone
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from narwhals import _plan as nwp
from narwhals._plan import Selector, selectors as ncs
from narwhals._plan._guards import is_expr, is_selector
from narwhals._utils import zip_strict
from narwhals.exceptions import ColumnNotFoundError, DuplicateError, InvalidOperationError
from tests.plan.utils import (
    Frame,
    assert_expr_ir_equal,
    assert_not_selector,
    is_expr_ir_equal,
    named_ir,
    re_compile,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from narwhals._plan.typing import IntoExpr, OperatorFn
    from narwhals.dtypes import DType


@pytest.fixture(scope="module")
def schema_nested_1() -> nw.Schema:
    return nw.Schema(
        {
            "a": nw.Int32(),
            "b": nw.List(nw.Int32()),
            "c": nw.List(nw.UInt32),
            "d": nw.Array(nw.Int32(), 3),
            "e": nw.List(nw.String),
            "f": nw.Struct({"x": nw.Int32()}),
        }
    )


@pytest.fixture(scope="module")
def schema_nested_2() -> nw.Schema:
    return nw.Schema(
        {
            "a": nw.Int32(),
            "b": nw.Array(nw.Int32(), 4),
            "c": nw.Array(nw.UInt32(), 4),
            "d": nw.Array(nw.Int32, 3),
            "e": nw.List(nw.Int32()),
            "f": nw.Array(nw.String(), 4),
            "g": nw.Struct({"x": nw.Int32()}),
        }
    )


@pytest.fixture(scope="module")
def schema_non_nested() -> nw.Schema:
    return nw.Schema(
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


@pytest.fixture(scope="module")
def schema_mixed() -> nw.Schema:
    return nw.Schema(
        {
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
            "u": nw.Struct({"a": nw.Int64(), "k": nw.String}),
        }
    )


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

    selector = ~(
        ncs.integer() | ncs.by_dtype(nw.Date(), nw.Datetime, nw.Duration, nw.Time())
    )
    df.assert_selects(selector, "cde", "def", "eee", "fgg", "qqR")

    selector = ncs.by_dtype(nw.Datetime("ns"), nw.Float32(), nw.UInt32, nw.Date)
    df.assert_selects(selector, "bbb", "def", "JJK")
    selector = ncs.by_dtype(
        nw.Int64,
        nw.Int128,
        nw.Duration("ns"),
        nw.Int8,
        nw.Binary(),
        nw.Int32(),
        nw.String,
    )
    expected = "ncs.by_dtype([Binary, Duration(time_unit='ns'), Int128, Int32, Int64, Int8, String])"
    assert_expr_ir_equal(selector, expected)


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


@pytest.mark.parametrize(
    ("dtypes", "expected"),
    [
        (
            [
                nw.Datetime,
                nw.Enum,
                nw.Datetime("s"),
                nw.Duration,
                nw.Struct,
                nw.List,
                nw.Array,
            ],
            ["l", "o", "q", "r", "s", "u"],
        ),
        ([nw.String(), nw.Boolean], ["k", "m"]),
        ([nw.Datetime("ms"), nw.Date, nw.List(nw.String)], ["n", "s"]),
        (
            [
                nw.Enum(["A", "B", "c"]),
                nw.Struct({"a": nw.List(nw.Int64), "k": nw.String}),
            ],
            [],
        ),
    ],
)
def test_selector_by_dtype_mixed(
    schema_mixed: nw.Schema,
    dtypes: Iterable[DType | type[DType]],
    expected: Iterable[str],
) -> None:
    df = Frame(schema_mixed)
    df.assert_selects(ncs.by_dtype(*dtypes), *expected)
    df.assert_selects(ncs.by_dtype(dtypes), *expected)


def test_selector_by_dtype_invalid_input() -> None:
    with pytest.raises(TypeError):
        ncs.by_dtype(999)  # type: ignore[arg-type]


def test_selector_by_index(schema_non_nested: nw.Schema) -> None:
    df = Frame(schema_non_nested)

    # # one or more positive indices
    df.assert_selects(ncs.by_index(0), "abc")
    df.assert_selects(ncs.first(), "abc")
    df.assert_selects(nwp.nth(0, 1, 2), "abc", "bbb", "cde")
    df.assert_selects(ncs.by_index(0, 1, 2), "abc", "bbb", "cde")

    # one or more negative indices
    df.assert_selects(ncs.by_index(-1), "qqR")
    df.assert_selects(ncs.last(), "qqR")
    df.assert_selects(ncs.by_index(-2, -1), "opp", "qqR")

    # range objects
    df.assert_selects(ncs.by_index(range(3)), "abc", "bbb", "cde")

    # exclude by index
    df.assert_selects(
        ~ncs.by_index(range(0, df.width, 2)), "bbb", "def", "fgg", "JJK", "opp"
    )

    df.assert_selects(ncs.by_index(0, 999, require_all=False), "abc")
    df.assert_selects(ncs.by_index(-1, -999, require_all=False), "qqR")
    df.assert_selects(ncs.by_index(1234, 5678, require_all=False))


def test_selector_by_index_invalid_input() -> None:
    with pytest.raises(TypeError):
        ncs.by_index("one")  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        ncs.by_index(["two", "three"])  # type: ignore[list-item]


def test_selector_by_index_not_found(schema_non_nested: nw.Schema) -> None:
    df = Frame(schema_non_nested)

    with pytest.raises(ColumnNotFoundError):
        df.project(ncs.by_index(999))

    df.assert_selects(ncs.by_index(999, -50, require_all=False))

    df = Frame(nw.Schema())
    df.assert_selects(ncs.by_index(111, -112, require_all=False))


def test_selector_by_index_reordering(schema_non_nested: nw.Schema) -> None:
    df = Frame(schema_non_nested)

    df.assert_selects(ncs.by_index(-3, -2, -1), "Lmn", "opp", "qqR")
    df.assert_selects(ncs.by_index(range(-3, 0)), "Lmn", "opp", "qqR")
    df.assert_selects(
        ncs.by_index(-3, 999, -2, -1, -48, require_all=False), "Lmn", "opp", "qqR"
    )


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


def test_selector_by_name_not_found(schema_non_nested: nw.Schema) -> None:
    df = Frame(schema_non_nested)

    with pytest.raises(ColumnNotFoundError):
        df.project(ncs.by_name("xxx", "fgg", "!!!"))

    with pytest.raises(ColumnNotFoundError):
        df.project(ncs.by_name("stroopwafel"))


def test_selector_by_name_invalid_input() -> None:
    with pytest.raises(TypeError):
        ncs.by_name(999)  # type: ignore[arg-type]


def test_selector_first_last(schema_non_nested: nw.Schema) -> None:
    df = Frame(schema_non_nested)
    first_name = "abc"
    mid_names = "bbb", "cde", "def", "eee", "fgg", "ghi", "JJK", "Lmn", "opp"
    last_name = "qqR"

    df.assert_selects(ncs.first(), first_name)
    df.assert_selects(~ncs.first(), *mid_names, last_name)
    df.assert_selects(ncs.last(), last_name)
    df.assert_selects(~ncs.last(), first_name, *mid_names)
    df.assert_selects(ncs.last() | ncs.first(), first_name, last_name)

    assert_expr_ir_equal(ncs.first(), "ncs.first()")
    assert_expr_ir_equal(ncs.last(), "ncs.last()")
    assert_expr_ir_equal(ncs.by_index(0), "ncs.first()")
    assert_expr_ir_equal(ncs.by_index(-1), "ncs.last()")

    repr_other = repr(ncs.by_index(1))
    assert "ncs.by_index(" in repr_other
    assert repr_other == repr_other.replace("ncs.first", "").replace("ncs.last", "")


def test_selector_datetime(schema_non_nested: nw.Schema) -> None:
    df = Frame(schema_non_nested)
    df.assert_selects(ncs.datetime(), "opp")
    df.assert_selects(ncs.datetime("ns"))
    all_columns = list(df.columns)
    all_columns.remove("opp")
    df.assert_selects(~ncs.datetime(), *all_columns)


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        (ncs.datetime(), ("d1", "d2", "d3", "d4", "d5")),
        (~ncs.datetime(), ()),
        (ncs.datetime(["ms", "ns"]), ("d1", "d2", "d5")),
        (ncs.datetime(["ms", "ns"], time_zone="*"), ("d1", "d2")),
        (~ncs.datetime(["ms", "ns"]), ("d3", "d4")),
        (~ncs.datetime(["ms", "ns"], time_zone="*"), ("d3", "d4", "d5")),
        (
            ncs.datetime(time_zone=["UTC", "Asia/Tokyo", "Europe/London"]),
            ("d1", "d2", "d3"),
        ),
        (ncs.datetime(time_zone="*"), ("d1", "d2", "d3")),
        (ncs.datetime("ns", time_zone="*"), ("d1", "d2")),
        (ncs.datetime(time_zone="UTC"), ("d2", "d3")),
        (ncs.datetime("us", time_zone="UTC"), ("d3",)),
        (ncs.datetime(time_zone="Asia/Tokyo"), ("d1",)),
        (ncs.datetime("us", time_zone="Asia/Tokyo"), ()),
        (ncs.datetime(time_zone=None), ("d4", "d5")),
        (ncs.datetime("ns", time_zone=None), ()),
        (~ncs.datetime(time_zone="*"), ("d4", "d5")),
        (~ncs.datetime("ns", time_zone="*"), ("d3", "d4", "d5")),
        (~ncs.datetime(time_zone="UTC"), ("d1", "d4", "d5")),
        (~ncs.datetime("us", time_zone="UTC"), ("d1", "d2", "d4", "d5")),
        (~ncs.datetime(time_zone="Asia/Tokyo"), ("d2", "d3", "d4", "d5")),
        (~ncs.datetime("us", time_zone="Asia/Tokyo"), ("d1", "d2", "d3", "d4", "d5")),
        (~ncs.datetime(time_zone=None), ("d1", "d2", "d3")),
        (~ncs.datetime("ns", time_zone=None), ("d1", "d2", "d3", "d4", "d5")),
        (ncs.datetime("ns"), ("d1", "d2")),
        (ncs.datetime("us"), ("d3", "d4")),
        (ncs.datetime("ms"), ("d5",)),
    ],
)
def test_selector_datetime_exhaustive(
    df_datetime: Frame, selector: Selector, expected: tuple[str, ...]
) -> None:
    df = df_datetime
    df.assert_selects(selector, *expected)


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


def test_selector_matches(schema_non_nested: nw.Schema) -> None:
    df = Frame(schema_non_nested)
    # NOTE: python's `re` raises on the original pattern this test used
    # > re.PatternError: global flags not at the start of the expression at position 1
    # https://github.com/pola-rs/polars/blob/84d66e960e3d462811f0575e0a6e4e78e34c618c/py-polars/tests/unit/test_selectors.py#L499
    pattern_str = r"(?i)[E-N]{3}"
    # We can get closer though, by accepting pre-compiled pattern
    pattern = re.compile(r"^[E-N]{3}$", re.IGNORECASE)
    positive = "eee", "fgg", "ghi", "JJK", "Lmn"
    negative = "abc", "bbb", "cde", "def", "opp", "qqR"

    df.assert_selects(ncs.matches(pattern_str), *positive)
    df.assert_selects(ncs.matches(pattern), *positive)

    df.assert_selects(~ncs.matches(pattern_str), *negative)
    df.assert_selects(~ncs.matches(pattern), *negative)


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


def test_selector_temporal(schema_non_nested: nw.Schema) -> None:
    df = Frame(schema_non_nested)
    positive = "ghi", "JJK", "Lmn", "opp"
    negative = "abc", "bbb", "cde", "def", "eee", "fgg", "qqR"
    df.assert_selects(ncs.temporal(), *positive)
    df.assert_selects(~ncs.temporal(), *negative)


def test_selector_float(schema_non_nested: nw.Schema) -> None:
    df = Frame(schema_non_nested)
    positive = "cde", "def"
    negative = "abc", "bbb", "eee", "fgg", "ghi", "JJK", "Lmn", "opp", "qqR"
    df.assert_selects(ncs.float(), *positive)
    df.assert_selects(~ncs.float(), *negative)


def test_selector_integer(schema_non_nested: nw.Schema) -> None:
    df = Frame(schema_non_nested)
    positive = "abc", "bbb"
    negative = "cde", "def", "eee", "fgg", "ghi", "JJK", "Lmn", "opp", "qqR"
    df.assert_selects(ncs.integer(), *positive)
    df.assert_selects(~ncs.integer(), *negative)


def test_selector_expansion() -> None:
    # https://github.com/pola-rs/polars/blob/84d66e960e3d462811f0575e0a6e4e78e34c618c/py-polars/tests/unit/test_selectors.py#L619
    df = Frame.from_names(*list("abcde"))

    s1 = nwp.all().meta.as_selector()
    s2 = nwp.col(["a", "b"]).meta.as_selector()
    s = s1 - s2
    df.assert_selects(s, "c", "d", "e")

    s1 = ncs.matches("^a|b$")
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

    with pytest.raises(
        InvalidOperationError, match=re_compile(r"cannot turn.+max.+into a selector")
    ):
        nwp.col("a").max().meta.as_selector()


def test_selector_set_ops(schema_non_nested: nw.Schema, schema_mixed: nw.Schema) -> None:
    df = Frame(schema_non_nested)

    temporal = ncs.temporal()

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
    selector = nwp.all().exclude("opp", "JJK").meta.as_selector() - (~temporal)
    df.assert_selects(selector, "ghi", "Lmn")

    sub_expr = ncs.matches("[yz]$") - nwp.col("colx")
    assert_not_selector(sub_expr)

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

    df = Frame(schema_mixed)
    selector = ~(ncs.numeric() | ncs.string())
    df.assert_selects(selector, "l", "m", "n", "o", "p", "q", "r", "s", "u")


def _is_binary_operator(function: OperatorFn) -> bool:
    return function in {operator.and_, operator.or_, operator.xor}


def _is_selector_operator(function: OperatorFn) -> bool:
    return function in {operator.and_, operator.or_, operator.xor, operator.sub}


@pytest.mark.parametrize(
    "arg_2",
    [1, nwp.col("a"), nwp.col("a").max(), ncs.numeric()],
    ids=["Scalar", "Column", "Expr", "Selector"],
)
@pytest.mark.parametrize(
    "function", [operator.and_, operator.or_, operator.xor, operator.add, operator.sub]
)
def test_selector_arith_binary_ops(
    arg_2: IntoExpr | Selector, function: OperatorFn
) -> None:
    # NOTE: These are the `polars.selectors` semantics
    # Parts of it may change with `polars>=2.0`, due to how confusing they are
    arg_1 = ncs.string()
    result_1 = function(arg_1, arg_2)
    if (
        _is_binary_operator(function)
        and is_expr(arg_2)
        and is_expr_ir_equal(arg_2, nwp.col("a"))
    ) or (_is_selector_operator(function) and is_selector(arg_2)):
        assert is_selector(result_1)
    else:
        assert_not_selector(result_1)

    if _is_binary_operator(function) and is_selector(arg_2):
        result_2 = function(arg_2, arg_1)
        assert is_selector(result_2)
    # `__sub__` is allowed, but `__rsub__` is not ...
    elif function is not operator.sub:
        result_2 = function(arg_2, arg_1)
        assert_not_selector(result_2)
    # ... unless both are `Selector`
    elif is_selector(arg_2):
        result_2 = function(arg_2, arg_1)
        assert is_selector(result_2)
    else:
        with pytest.raises(TypeError):
            function(arg_2, arg_1)


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


def test_selector_list(schema_nested_1: nw.Schema) -> None:
    df = Frame(schema_nested_1)

    # inner None
    df.assert_selects(ncs.list(), "b", "c", "e")
    # Inner All (as a DTypeSelector)
    df.assert_selects(ncs.list(ncs.all()), "b", "c", "e")
    # inner DTypeSelector
    df.assert_selects(ncs.list(ncs.integer()), "b", "c")
    df.assert_selects(ncs.list(inner=ncs.string()), "e")
    # inner BinarySelector
    df.assert_selects(
        ncs.list(ncs.by_dtype(nw.Int32) | ncs.by_dtype(nw.UInt32)), "b", "c"
    )
    # inner InvertSelector
    df.assert_selects(ncs.list(~ncs.all()))


def test_selector_array(schema_nested_2: nw.Schema) -> None:
    df = Frame(schema_nested_2)
    df.assert_selects(ncs.array(), "b", "c", "d", "f")
    df.assert_selects(ncs.array(ncs.all()), "b", "c", "d", "f")
    df.assert_selects(ncs.array(size=4), "b", "c", "f")
    df.assert_selects(ncs.array(inner=ncs.integer()), "b", "c", "d")
    df.assert_selects(ncs.array(inner=ncs.string()), "f")


def test_selector_non_dtype_inside_dtype(schema_nested_2: nw.Schema) -> None:
    df = Frame(schema_nested_2)

    with pytest.raises(
        TypeError, match=r"expected datatype based expression got.+by_name\("
    ):
        df.project(ncs.list(inner=ncs.by_name("???")))

    with pytest.raises(
        TypeError, match=r"expected datatype based expression got.+by_name\("
    ):
        df.project(ncs.array(inner=ncs.by_name("???")))


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


def test_selector_decimal(schema_mixed: nw.Schema) -> None:
    df = Frame(schema_mixed)
    df.assert_selects(ncs.decimal())
    df = df.from_mapping(
        {"zz0": nw.Float64(), "zz1": nw.Decimal(38, 5), "zz2": nw.Decimal()}
    )
    df.assert_selects(ncs.numeric(), "zz0", "zz1", "zz2")
    df.assert_selects(ncs.decimal(), "zz1", "zz2")
    df.assert_selects(~ncs.decimal(), "zz0")


def test_selector_matches_22816() -> None:
    df = Frame.from_names("ham", "hamburger", "foo", "bar")
    df.assert_selects(ncs.matches(r"^ham.*$"), "ham", "hamburger")
    df.assert_selects(ncs.matches(r".*burger"), "hamburger")


def test_selector_by_name_order_19384() -> None:
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


def test_name_suffix_complex_selector(schema_mixed: nw.Schema) -> None:
    df = Frame(schema_mixed)
    selector = (
        ncs.all() - (ncs.categorical() | ncs.by_name("a", "b") | ncs.matches("[fqohim]"))
        ^ ncs.by_name("u", "a", "b", "d", "e", "f", "g")
    ).name.suffix("_after")
    selected_names = "a", "b", "c", "f", "j", "k", "l", "n", "r", "s"
    expecteds = (named_ir(f"{name}_after", nwp.col(name)) for name in selected_names)
    actuals = df.project(selector)

    for actual, expected in zip_strict(actuals, expecteds):
        assert_expr_ir_equal(actual, expected)


def test_name_map_chain_21164() -> None:
    # https://github.com/pola-rs/polars/blob/5b90db75911c70010d0c0a6941046e6144af88d4/py-polars/tests/unit/operations/namespaces/test_name.py#L110-L115
    df = Frame.from_names("MyCol")
    aliased = nwp.col("MyCol").alias("mycol_suffix")
    rename_chain = ncs.all().name.to_lowercase().name.suffix("_suffix")
    df.assert_selects(aliased, "mycol_suffix")
    df.assert_selects(rename_chain, "mycol_suffix")


def test_when_then_keep_map_13858() -> None:
    # https://github.com/pola-rs/polars/blob/aaa11d6af7383a5f9b62f432e14cc2d4af6d8548/py-polars/tests/unit/operations/namespaces/test_name.py#L118-L138
    # https://github.com/pola-rs/polars/issues/13858
    df = Frame.from_names("a", "b")
    aliased = nwp.int_range(3).alias("b_other")
    when_keep_chain = (
        nwp.when(nwp.lit(True))
        .then(nwp.int_range(nwp.len()))
        .otherwise(1 + nwp.col("b"))
        .name.keep()
        .name.suffix("_other")
    )
    df.assert_selects(aliased, "b_other")
    df.assert_selects(when_keep_chain, "b_other")


def test_keep_name_struct_field_23669() -> None:
    df = Frame.from_mapping(
        {"foo": nw.Struct({"x": nw.Int64}), "bar": nw.Struct({"x": nw.Int64})}
    )

    with pytest.raises(DuplicateError):
        df.project(nwp.all().struct.field("x"))

    df.assert_selects(nwp.all().struct.field("x").name.keep(), "foo", "bar")
