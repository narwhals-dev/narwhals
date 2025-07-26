from __future__ import annotations

from datetime import datetime

import pytest

import narwhals as nw
from tests.utils import PYARROW_VERSION, Constructor, assert_equal_data


def test_unpivot(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "cudf" in str(constructor) or (
        "pyarrow_table" in str(constructor) and PYARROW_VERSION < (13, 0, 0)
    ):
        request.applymarker(pytest.mark.xfail)

    data = {
        "date": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        "aapl": [110, 100, 105],
        "tsla": [220, 200, 210],
        "msft": [330, 300, 315],
        "nflx": [420, 400, 440],
    }

    df = nw.from_native(constructor(data))

    result = df.unpivot(index="date", value_name="price").sort(by=["date", "variable"])

    expected = {
        "date": [
            *[datetime(2020, 1, 1)] * 4,
            *[datetime(2020, 1, 2)] * 4,
            *[datetime(2020, 1, 3)] * 4,
        ],
        "variable": [*["aapl", "msft", "nflx", "tsla"] * 3],
        "price": [110, 330, 420, 220, 100, 300, 400, 200, 105, 315, 440, 210],
    }
    assert_equal_data(result, expected)
