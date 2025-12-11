from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from narwhals._plan.typing import IntoExpr
    from tests.conftest import Data


@pytest.fixture(scope="module")
def data() -> Data:
    return {
        "a": [[2, 2, 3, None, None], None, [], [None]],
        "b": [[1, 2, 2], [3, 4], [5, 5, 5, 6], [7]],
        "c": [1, 3, None, 2],
    }


a = nwp.col("a")
b = nwp.col("b")


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (2, [True, None, False, False]),
        (4, [False, None, False, False]),
        (nwp.col("c").last() + 1, [True, None, False, False]),
        (nwp.lit(None, nw.Int32), [True, None, False, True]),
    ],
)
def test_list_contains(data: Data, item: IntoExpr, expected: list[bool | None]) -> None:
    df = dataframe(data).with_columns(a.cast(nw.List(nw.Int32)))
    result = df.select(a.list.contains(item))
    assert_equal_data(result, {"a": expected})
