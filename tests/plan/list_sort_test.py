from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from tests.conftest import Data
    from tests.plan.utils import SubList

ASC = False
DESC = True
NULLS_FIRST = False
NULLS_LAST = True


expected_desc_nulls_last = [
    [4, 3, 2, 2, -10, None, None],
    [-1],
    None,
    [None, None, None],
    [],
]
expected_desc_nulls_first = [
    [None, None, 4, 3, 2, 2, -10],
    [-1],
    None,
    [None, None, None],
    [],
]
expected_asc_nulls_last = [
    [-10, 2, 2, 3, 4, None, None],
    [-1],
    None,
    [None, None, None],
    [],
]
expected_asc_nulls_first = [
    [None, None, -10, 2, 2, 3, 4],
    [-1],
    None,
    [None, None, None],
    [],
]


@pytest.fixture(scope="module")
def data() -> Data:
    return {"a": [[3, 2, 2, 4, -10, None, None], [-1], None, [None, None, None], []]}


a = nwp.col("a")


@pytest.mark.xfail(reason="TODO: `ArrowExpr.list.sort`", raises=NotImplementedError)
@pytest.mark.parametrize(
    ("descending", "nulls_last", "expected"),
    [
        (DESC, NULLS_LAST, expected_desc_nulls_last),
        (DESC, NULLS_FIRST, expected_desc_nulls_first),
        (ASC, NULLS_LAST, expected_asc_nulls_last),
        (ASC, NULLS_FIRST, expected_asc_nulls_first),
    ],
)
def test_list_sort(
    data: Data, *, descending: bool, nulls_last: bool, expected: list[SubList[int]]
) -> None:  # pragma: no cover
    df = dataframe(data).with_columns(a.cast(nw.List(nw.Int32)))
    expr = a.list.sort(descending=descending, nulls_last=nulls_last)
    result = df.select(expr)
    assert_equal_data(result, {"a": expected})
