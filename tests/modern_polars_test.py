from __future__ import annotations

from datetime import date

import pytest

import narwhals as nw
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


def test_pivot(
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
) -> None:
    not_implemented = ("modin", "pyarrow_table_constructor")
    if any(lib for lib in not_implemented if lib in str(constructor_eager)):
        request.applymarker(pytest.mark.xfail)

    data = {
        "date": [
            *[date(2020, 1, 1)] * 4,
            *[date(2020, 1, 2)] * 4,
            *[date(2020, 1, 3)] * 4,
        ],
        "ticker": [*["AAPL", "TSLA", "MSFT", "NFLX"] * 3],
        "price": [100, 200, 300, 400, 110, 220, 330, 420, 105, 210, 315, 440],
    }
    df = nw.from_native(constructor_eager(data), eager_only=True)

    pivoted = df.pivot(index="date", values="price", on="ticker").sort("date")

    expected = {
        "date": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
        "AAPL": [100, 110, 105],
        "TSLA": [200, 220, 210],
        "MSFT": [300, 330, 315],
        "NFLX": [400, 420, 440],
    }

    assert_equal_data(pivoted, expected)
