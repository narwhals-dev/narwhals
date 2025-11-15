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


@pytest.fixture
def data() -> Data:
    return {"a": [1, 4, 2, 5], "b": [1, 0, 2, 0], "c": [None, "hi", "hello", "howdy"]}


@pytest.mark.xfail(reason="Not implemented `is_in_seq`", raises=NotImplementedError)
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
    ],
)
def test_expr_is_in(
    data: Data, expr: nwp.Expr, expected: Data
) -> None:  # pragma: no cover
    result = dataframe(data).select(expr)
    assert_equal_data(result, expected)


@pytest.mark.xfail(reason="Not implemented `Series.is_in`", raises=NotImplementedError)
@pytest.mark.parametrize(
    ("column", "other", "expected"), [("a", [4, 5], [False, True, False, True])]
)
def test_ser_is_in(
    data: Data,
    column: Literal["a", "b", "c"],
    other: Iterable[Any],
    expected: Sequence[Any],
) -> None:  # pragma: no cover
    result = series(data[column]).alias(column).is_in(other)
    assert_equal_series(result, expected, column)


def test_is_in_other(data: Data) -> None:
    df = dataframe(data)
    with pytest.raises(TypeError, match=r"is_in.+doesn't accept.+str"):
        df.with_columns(contains=nwp.col("a").is_in("sets"))


@pytest.mark.xfail(reason="Not implemented `is_in_series`", raises=NotImplementedError)
def test_filter_is_in_with_series(data: Data) -> None:  # pragma: no cover
    df = dataframe(data)
    expr = nwp.col("a").is_in(df.get_column("b"))
    result = df.filter(expr)
    expected = {"a": [1, 2], "b": [1, 2]}
    assert_equal_data(result, expected)
