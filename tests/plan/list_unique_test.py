from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
import narwhals._plan as nwp
from tests.plan.utils import assert_equal_series, dataframe

if TYPE_CHECKING:
    from tests.conftest import Data


@pytest.fixture(scope="module")
def data() -> Data:
    return {
        "a": [[2, 2, 3, None, None], None, [], [None]],
        "b": [[1, 2, 2], [3, 4], [5, 5, 5, 6], [7]],
    }


def test_list_unique(data: Data) -> None:
    df = dataframe(data).select(nwp.col("a").cast(nw.List(nw.Int32)))
    ser = df.select(nwp.col("a").list.unique()).to_series()
    result = ser.to_list()
    assert len(result) == 4
    assert len(result[0]) == 3
    assert set(result[0]) == {2, 3, None}
    assert result[1] is None
    assert len(result[2]) == 0
    assert len(result[3]) == 1

    assert_equal_series(ser.explode(), [2, 3, None, None, None, None], "a")


# TODO @dangotbanned: Report `ListScalar.values` bug upstream
# - Returning `None` breaks: `__len__`,` __getitem__`, `__iter__`
# - Which breaks `pa.array([<pyarrow.ListScalar: None>], pa.list_(pa.int64()))`
@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ([None, "A", "B", "A", "A", "B"], [None, "A", "B"]),
        (None, None),
        ([], []),
        ([None], [None]),
    ],
)
def test_list_unique_scalar(
    row: list[str | None] | None, expected: list[str | None] | None
) -> None:
    data = {"a": [row]}
    df = dataframe(data).select(nwp.col("a").cast(nw.List(nw.String)).first())
    result = df.select(nwp.col("a").list.unique()).to_series()
    assert_equal_series(result, [expected], "a")


def test_list_unique_all_valid(data: Data) -> None:
    df = dataframe(data).select(nwp.col("b").cast(nw.List(nw.Int32)))
    ser = df.select(nwp.col("b").list.unique()).to_series()
    result = ser.to_list()
    assert set(result[0]) == {1, 2}
    assert set(result[1]) == {3, 4}
    assert set(result[2]) == {5, 6}
    assert set(result[3]) == {7}
