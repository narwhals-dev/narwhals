from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, Literal

import pytest

import narwhals as nw
import narwhals._plan as nwp
from narwhals._plan import selectors as ncs
from tests.plan.utils import DataFrame, Series, assert_equal_data, assert_equal_series

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from tests.conftest import Data

pytest.importorskip("pyarrow")

import pyarrow as pa


@pytest.fixture
def data() -> Data:
    return {"a": [1, 4, 2, 5], "b": [1, 0, 2, 0], "c": [None, "hi", "hello", "howdy"]}


# TODO @dangotbanned: Address polars deprecation compat (applies to all tests)
# https://github.com/pola-rs/polars/pull/22178
@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nwp.col("a").is_in([4, 5]), {"a": [False, True, False, True]}),
        (nwp.col("a").is_in([]), {"a": [False, False, False, False]}),
        (nwp.col("b").is_in(deque([0, 1])), {"b": [True, True, False, True]}),
        (
            ncs.integer().is_in([5, 6, 0]),
            {"a": [False, False, False, True], "b": [False, True, False, True]},
        ),
        (
            (nwp.col("b").max() + nwp.col("a")).is_in(range(5, 10)),
            {"b": [False, True, False, True]},
        ),
        (ncs.string().last().is_in(iter(["howdy"])), {"c": [True]}),
    ],
)
def test_expr_is_in_seq(
    data: Data, expr: nwp.Expr, expected: Data, dataframe: DataFrame
) -> None:
    result = dataframe(data).select(expr)
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nwp.col("a").is_in(nwp.col("b")), {"a": [True, False, True, False]}),
        (nwp.col("a").is_in(nwp.nth(1)), {"a": [True, False, True, False]}),
        (nwp.col("b").is_in(nwp.col("a") - 5), {"b": [False, True, False, True]}),
        (
            (nwp.col("b").max() + nwp.col("a")).is_in(nwp.int_range(5, 10)),
            {"b": [False, True, False, True]},
        ),
        (nwp.col("a").last().is_in(ncs.first()), {"a": [True]}),
        (
            (nwp.col("a").last() - nwp.col("b").first()).is_in(
                ncs.integer() - ncs.first()
            ),
            {"a": [False]},
        ),
        (
            (ncs.integer() + 4).is_in(nwp.nth(0).filter(nwp.col("b") < 2)),
            {"a": [True, False, False, False], "b": [True, True, False, True]},
        ),
    ],
)
def test_expr_is_in_expr(
    data: Data, expr: nwp.Expr, expected: Data, dataframe: DataFrame
) -> None:
    df = dataframe(data)
    assert_equal_data(df.select(expr), expected)


@pytest.mark.parametrize(
    ("column", "other", "expected"),
    [
        ("a", [4, 5], [False, True, False, True]),
        ("a", [], [False, False, False, False]),
        ("b", deque([0, 1]), [True, True, False, True]),
        ("b", [2], [False, False, True, False]),
    ],
)
@pytest.mark.parametrize("use_series", [True, False])
def test_ser_is_in_seq(
    data: Data,
    column: Literal["a", "b"],
    other: Iterable[Any],
    expected: Sequence[Any],
    series: Series,
    use_series: bool,  # noqa: FBT001
) -> None:
    ser = series(data[column], name=column)
    if use_series:
        other = ser.from_iterable(other, backend=series.implementation)
    result = ser.is_in(other)
    assert_equal_series(result, expected, column)


def test_expr_is_in_series(data: Data, dataframe: DataFrame) -> None:
    df = dataframe(data)

    a = nwp.col("a")
    a_first = a.first()
    a_last = a.last()
    a_ser = df.get_column("a")
    b_ser = df.get_column("b")

    assert_equal_data(df.filter(a.is_in(b_ser)), {"a": [1, 2], "b": [1, 2]})
    assert_equal_data(df.select(a_last.is_in(b_ser)), {"a": [False]})
    assert_equal_data(df.select(a_first.is_in(b_ser)), {"a": [True]})
    assert_equal_data(df.select((a_last - a_first).is_in(a_ser)), {"a": [True]})
    assert_equal_data(df.select((a_last - a_first).is_in(b_ser)), {"a": [False]})


# TODO @dangotbanned: Address polars compat (nulls_equal)
# https://github.com/pola-rs/polars/pull/21426
def test_expr_is_in_seq_nulls(
    data: Data, dataframe: DataFrame, request: pytest.FixtureRequest
) -> None:
    expr = nwp.col("c").is_in(("howdy", None))
    expected = {"c": [True, False, False, True]}
    dataframe.xfail(
        request,
        dataframe.is_polars(),
        reason=(
            "TODO @dangotbanned: Add `nulls_equal` parameter to `is_in`.\n"
            "https://github.com/pola-rs/polars/pull/21426"
        ),
        raises=AssertionError,
    )
    result = dataframe(data).select(expr)
    assert_equal_data(result, expected)


def test_expr_is_in_expr_nulls(
    data: Data, dataframe: DataFrame, request: pytest.FixtureRequest
) -> None:
    expr = ncs.string().is_in(nwp.lit(None, nw.String))
    expected = {"c": [True, False, False, False]}
    dataframe.xfail(
        request,
        dataframe.is_polars(),
        reason=(
            "TODO @dangotbanned: Add `nulls_equal` parameter to `is_in`.\n"
            "https://github.com/pola-rs/polars/pull/21426"
        ),
        raises=AssertionError,
    )
    df = dataframe(data)
    assert_equal_data(df.select(expr), expected)


@pytest.mark.parametrize(
    ("other", "expected"),
    [
        (("howdy", None), [True, False, False, True]),
        (pa.array(["hi", "hello"]), [False, True, True, False]),
    ],
)
@pytest.mark.parametrize("use_series", [True, False])
def test_ser_is_in_nulls(
    data: Data,
    other: Iterable[Any],
    expected: Sequence[Any],
    series: Series,
    use_series: bool,  # noqa: FBT001
    request: pytest.FixtureRequest,
) -> None:
    series.xfail(
        request,
        series.is_polars(),
        reason=(
            "TODO @dangotbanned: Add `nulls_equal` parameter to `is_in`.\n"
            "https://github.com/pola-rs/polars/pull/21426"
        ),
        raises=AssertionError,
    )
    column = "c"
    ser = series(data[column], name=column)
    if use_series:
        other = ser.from_iterable(other, backend=series.implementation)
    result = ser.is_in(other)
    assert_equal_series(result, expected, column)
