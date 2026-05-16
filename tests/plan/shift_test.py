from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe

if TYPE_CHECKING:
    from tests.conftest import Data


@pytest.fixture(scope="module")
def data() -> Data:
    return {
        "i": [None, 1, 2, 3, 4],
        "a": [0, 1, 2, 3, 4],
        "b": [1, 2, 3, 5, 3],
        "c": [5, 4, 3, 2, 1],
    }


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (2, {"a": [0, 1, 2], "b": [1, 2, 3], "c": [5, 4, 3]}),
        (4, {"a": [None, None, 0], "b": [None, None, 1], "c": [None, None, 5]}),
        (-3, {"a": [None, None, None], "b": [None, None, None], "c": [None, None, None]}),
    ],
)
def test_shift(data: Data, n: int, expected: Data) -> None:
    df = dataframe(data)
    result = df.with_columns(nwp.col("a", "b", "c").shift(n).over(order_by="i")).filter(
        nwp.col("i") > 1
    )
    assert_equal_data(result, {"i": [2, 3, 4], **expected})


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, [None, 1, 2, 3, 1, 2, 3, 1, 2]),
        (-1, [2, 3, 1, 2, 3, 1, 2, 3, None]),
        (0, [1, 2, 3, 1, 2, 3, 1, 2, 3]),
    ],
)
def test_shift_multi_chunk_pyarrow(n: int, expected: list[float | None]) -> None:
    pytest.importorskip("pyarrow")
    df = nwp.DataFrame.from_dict({"a": [1, 2, 3]}, backend="pyarrow")
    df = nwp.concat([df, df, df])
    assert df.to_native().column("a").num_chunks != 1, "Need multiple chunks."
    assert_equal_data(df.select(nwp.col("a").shift(n)), {"a": expected})
