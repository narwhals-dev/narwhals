from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

import pytest

import narwhals._plan.selectors as ncs
from narwhals.exceptions import ColumnNotFoundError
from tests.plan.utils import DataFrame, assert_equal_data, re_compile

if TYPE_CHECKING:
    from narwhals._plan.typing import ColumnNameOrSelector, OneOrIterable


def test_with_row_index_eager(dataframe: DataFrame) -> None:
    data = {"abc": ["foo", "bars"], "xyz": [100, 200], "const": [42, 42]}
    result = dataframe(data).with_row_index()
    expected = {"index": [0, 1], **data}
    assert_equal_data(result, expected)


IDX_1: Final = [0, 2, 1]
IDX_2: Final = [2, 0, 1]
IDX_3: Final = [1, 2, 0]
first, last = ncs.first(), ncs.last()


@pytest.mark.parametrize(
    ("order_by", "expected_index"),
    [
        pytest.param(["a"], IDX_1, id="['a']"),
        pytest.param(first, IDX_1, id="first()"),
        pytest.param(ncs.string(), IDX_1, id="string()"),
        pytest.param(["c"], IDX_2, id="['c']"),
        pytest.param(last, IDX_2, id="last()"),
        pytest.param(
            ncs.integer() - ncs.by_index(1), IDX_2, id="integer() - by_index(1)"
        ),
        pytest.param(["a", "c"], IDX_3, id="['a', 'c']"),
        pytest.param([first, "c"], IDX_3, id="[first(), 'c']"),
        pytest.param(["a", ncs.by_name("c")], IDX_3, id="['a', by_name('c')]"),
        pytest.param(["c", "a"], IDX_2, id="['c', 'a']"),
        pytest.param([ncs.by_index(-1, 0)], IDX_2, id="[by_index(-1, 0)]"),
        pytest.param([last, first], IDX_2, id="[last(), first()]"),
    ],
)
def test_with_row_index_by(
    dataframe: DataFrame,
    order_by: OneOrIterable[ColumnNameOrSelector],
    expected_index: list[int],
) -> None:
    # https://github.com/narwhals-dev/narwhals/issues/3289
    data = {"a": ["A", "B", "A"], "b": [1, 2, 3], "c": [9, 2, 4]}

    result = dataframe(data).with_row_index(name="index", order_by=order_by).sort("b")
    expected = {"index": expected_index, **data}
    assert_equal_data(result, expected)


def test_with_row_index_by_invalid(dataframe: DataFrame) -> None:
    data = {"a": ["A", "B", "A"], "b": [1, 2, 3], "c": [9, 2, 4]}
    df = dataframe(data)

    with pytest.raises(ColumnNotFoundError, match=re.escape("not found: ['d']")):
        df.with_row_index(order_by="d")

    with pytest.raises(ColumnNotFoundError, match=re.escape("not found: ['e']")):
        df.with_row_index(order_by=["e", "b"])

    with pytest.raises(ColumnNotFoundError, match=r"Invalid column index 5"):
        df.with_row_index(order_by=ncs.by_index(5))


def test_with_row_index_by_empty_selection(dataframe: DataFrame) -> None:
    data = {"a": ["A", "B", "A"], "b": [1, 2, 3], "c": [9, 2, 4]}
    df = dataframe(data)
    with pytest.raises(ColumnNotFoundError, match=re.escape("ncs.datetime(")):
        df.with_row_index(order_by=ncs.datetime())

    schema = re.escape("{'a': String, 'b': Int64, 'c': Int64}")
    pattern = re_compile(rf"ncs.float\(\).*ncs.temporal\(\).*Hint:.+{schema}")
    with pytest.raises(ColumnNotFoundError, match=pattern):
        df.with_row_index(order_by=[ncs.float(), ncs.temporal()])
