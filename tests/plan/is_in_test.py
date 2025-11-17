from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, Literal

import pytest

import narwhals._plan as nwp
from narwhals._plan import selectors as ncs
from tests.plan.utils import assert_equal_data, assert_equal_series, dataframe, series

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from tests.conftest import Data

pytest.importorskip("pyarrow")

import pyarrow as pa


@pytest.fixture
def data() -> Data:
    return {"a": [1, 4, 2, 5], "b": [1, 0, 2, 0], "c": [None, "hi", "hello", "howdy"]}


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nwp.col("a").is_in([4, 5]), {"a": [False, True, False, True]}),
        (nwp.col("a").is_in([]), {"a": [False, False, False, False]}),
        (nwp.col("b").is_in(deque([0, 1])), {"b": [True, True, False, True]}),
        (nwp.col("c").is_in(("howdy", None)), {"c": [True, False, False, True]}),
        (
            ncs.integer().is_in([5, 6, 0]),
            {"a": [False, False, False, True], "b": [False, True, False, True]},
        ),
        (ncs.string().last().is_in(iter(["howdy"])), {"c": [True]}),
        (
            (nwp.col("b").max() + nwp.col("a")).is_in(range(5, 10)),
            {"b": [False, True, False, True]},
        ),
    ],
)
def test_expr_is_in(data: Data, expr: nwp.Expr, expected: Data) -> None:
    result = dataframe(data).select(expr)
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("column", "other", "expected"),
    [
        ("a", [4, 5], [False, True, False, True]),
        ("a", [], [False, False, False, False]),
        ("b", deque([0, 1]), [True, True, False, True]),
        ("c", ("howdy", None), [True, False, False, True]),
        ("b", series([2]), [False, False, True, False]),
        ("c", pa.array(["hi", "hello"]), [False, True, True, False]),
    ],
)
def test_ser_is_in(
    data: Data,
    column: Literal["a", "b", "c"],
    other: Iterable[Any],
    expected: Sequence[Any],
) -> None:
    result = series(data[column]).alias(column).is_in(other)
    assert_equal_series(result, expected, column)


def test_is_in_other(data: Data) -> None:
    df = dataframe(data)
    with pytest.raises(TypeError, match=r"is_in.+doesn't accept.+str"):
        df.with_columns(contains=nwp.col("a").is_in("sets"))


def test_expr_is_in_series(data: Data) -> None:
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
