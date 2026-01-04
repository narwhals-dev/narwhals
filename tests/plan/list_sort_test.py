from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

import narwhals as nw
import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tests.conftest import Data
    from tests.plan.utils import SubList


@pytest.fixture(scope="module")
def data() -> Data:
    return {
        "a": [
            [3, 2, 2, None, 4, -10, None, None],
            [-1],
            None,
            [None, None, None],
            [],
            [None, 0, 0, None, -5, 10, None, 0, None],
            [None],
        ]
    }


ASC = False
DESC = True
NULLS_FIRST = False
NULLS_LAST = True


EXPECTED: Final[Mapping[tuple[bool, bool], list[SubList[int]]]] = {
    (DESC, NULLS_LAST): [
        [4, 3, 2, 2, -10, None, None, None],
        [-1],
        None,
        [None, None, None],
        [],
        [10, 0, 0, 0, -5, None, None, None, None],
        [None],
    ],
    (DESC, NULLS_FIRST): [
        [None, None, None, 4, 3, 2, 2, -10],
        [-1],
        None,
        [None, None, None],
        [],
        [None, None, None, None, 10, 0, 0, 0, -5],
        [None],
    ],
    (ASC, NULLS_LAST): [
        [-10, 2, 2, 3, 4, None, None, None],
        [-1],
        None,
        [None, None, None],
        [],
        [-5, 0, 0, 0, 10, None, None, None, None],
        [None],
    ],
    (ASC, NULLS_FIRST): [
        [None, None, None, -10, 2, 2, 3, 4],
        [-1],
        None,
        [None, None, None],
        [],
        [None, None, None, None, -5, 0, 0, 0, 10],
        [None],
    ],
}


a = nwp.col("a")
cast_a = a.cast(nw.List(nw.Int32))

sort_options = pytest.mark.parametrize(
    ("descending", "nulls_last"),
    [(DESC, NULLS_LAST), (DESC, NULLS_FIRST), (ASC, NULLS_LAST), (ASC, NULLS_FIRST)],
    ids=["desc-nulls-last", "desc-nulls-first", "asc-nulls-last", "asc-nulls-first"],
)


@sort_options
def test_list_sort(data: Data, *, descending: bool, nulls_last: bool) -> None:
    df = dataframe(data).with_columns(cast_a)
    expr = a.list.sort(descending=descending, nulls_last=nulls_last)
    result = df.select(expr)
    expected = {"a": EXPECTED[(descending, nulls_last)]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize("row_index", range(7))
@sort_options
def test_list_sort_scalar(
    data: Data, row_index: int, *, descending: bool, nulls_last: bool
) -> None:
    row = data["a"][row_index]
    df = dataframe({"a": [row]}).with_columns(cast_a)
    expr = a.first().list.sort(descending=descending, nulls_last=nulls_last)
    result = df.select(expr)
    expected = {"a": [EXPECTED[(descending, nulls_last)][row_index]]}
    assert_equal_data(result, expected)
