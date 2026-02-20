from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

import narwhals._plan.selectors as ncs
from narwhals.exceptions import ColumnNotFoundError
from tests.plan.utils import assert_equal_data, dataframe, re_compile

if TYPE_CHECKING:
    from narwhals._plan.typing import ColumnNameOrSelector, OneOrIterable


def test_with_row_index_eager() -> None:
    data = {"abc": ["foo", "bars"], "xyz": [100, 200], "const": [42, 42]}
    result = dataframe(data).with_row_index()
    expected = {"index": [0, 1], **data}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("order_by", "expected_index"),
    [
        (["a"], [0, 2, 1]),
        (ncs.first(), [0, 2, 1]),
        (ncs.string(), [0, 2, 1]),
        (["c"], [2, 0, 1]),
        (ncs.last(), [2, 0, 1]),
        (ncs.integer() - ncs.by_index(1), [2, 0, 1]),
        (["a", "c"], [1, 2, 0]),
        ([ncs.first(), "c"], [1, 2, 0]),
        (["a", ncs.by_name("c")], [1, 2, 0]),
        (["c", "a"], [2, 0, 1]),
        ([ncs.by_index(-1, 0)], [2, 0, 1]),
        ([ncs.last(), ncs.first()], [2, 0, 1]),
    ],
)
def test_with_row_index_by(
    order_by: OneOrIterable[ColumnNameOrSelector], expected_index: list[int]
) -> None:
    # https://github.com/narwhals-dev/narwhals/issues/3289
    data = {"a": ["A", "B", "A"], "b": [1, 2, 3], "c": [9, 2, 4]}
    result = dataframe(data).with_row_index(name="index", order_by=order_by).sort("b")
    expected = {"index": expected_index, **data}
    assert_equal_data(result, expected)


def test_with_row_index_by_invalid() -> None:
    data = {"a": ["A", "B", "A"], "b": [1, 2, 3], "c": [9, 2, 4]}
    df = dataframe(data)

    with pytest.raises(ColumnNotFoundError, match=re.escape("not found: ['d']")):
        df.with_row_index(order_by="d")

    with pytest.raises(ColumnNotFoundError, match=re.escape("not found: ['e']")):
        df.with_row_index(order_by=["e", "b"])

    with pytest.raises(ColumnNotFoundError, match=r"Invalid column index 5"):
        df.with_row_index(order_by=ncs.by_index(5))


def test_with_row_index_by_empty_selection() -> None:
    data = {"a": ["A", "B", "A"], "b": [1, 2, 3], "c": [9, 2, 4]}
    df = dataframe(data)
    with pytest.raises(ColumnNotFoundError, match=re.escape("ncs.datetime(")):
        df.with_row_index(order_by=ncs.datetime())

    schema = re.escape("{'a': String, 'b': Int64, 'c': Int64}")
    pattern = re_compile(rf"ncs.float\(\).*ncs.temporal\(\).*Hint:.+{schema}")
    with pytest.raises(ColumnNotFoundError, match=pattern):
        df.with_row_index(order_by=[ncs.float(), ncs.temporal()])
