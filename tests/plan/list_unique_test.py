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
    return {"a": [[2, 2, 3, None, None], None, [], [None]]}


def test_list_unique(data: Data) -> None:
    df = dataframe(data).with_columns(nwp.col("a"))
    ser = df.select(nwp.col("a").cast(nw.List(nw.Int32)).list.unique()).to_series()
    result = ser.to_list()
    assert len(result) == 4
    assert len(result[0]) == 3
    assert set(result[0]) == {2, 3, None}
    assert result[1] is None
    assert len(result[2]) == 0
    assert len(result[3]) == 1

    assert_equal_series(ser.explode(), [2, 3, None, None, None, None], "a")
