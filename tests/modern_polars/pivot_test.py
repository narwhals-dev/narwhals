from __future__ import annotations

from datetime import datetime

import pytest

import narwhals as nw
from tests.utils import POLARS_VERSION, ConstructorEager, assert_equal_data


def test_pivot(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if any(x in str(constructor_eager) for x in ("pyarrow_table", "modin")):
        request.applymarker(pytest.mark.xfail)
    if "polars" in str(constructor_eager) and POLARS_VERSION < (1, 0):
        pytest.skip()

    data = {
        "date": [
            *[datetime(2020, 1, 2)] * 4,
            *[datetime(2020, 1, 1)] * 4,
            *[datetime(2020, 1, 3)] * 4,
        ],
        "ticker": [*["AAPL", "TSLA", "MSFT", "NFLX"] * 3],
        "price": [100, 200, 300, 400, 110, 220, 330, 420, 105, 210, 315, 440],
    }
    df = nw.from_native(constructor_eager(data), eager_only=True).sort("date")

    pivoted = df.pivot(index="date", values="price", on="ticker")

    expected = {
        "date": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        "AAPL": [110, 100, 105],
        "TSLA": [220, 200, 210],
        "MSFT": [330, 300, 315],
        "NFLX": [420, 400, 440],
    }

    assert_equal_data(pivoted, expected)
