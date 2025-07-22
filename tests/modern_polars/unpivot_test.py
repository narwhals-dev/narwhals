from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from datetime import date

import pytest

import narwhals as nw
from tests.utils import PYARROW_VERSION, Constructor, assert_equal_data


def test_unpivot(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    context = (  # a. sqlframe is expected to be case-insensitive?, b. dask dates are in string-format
        pytest.raises(AssertionError)
        if ("sqlframe" in str(constructor) or "dask" in str(constructor))
        else does_not_raise()
    )

    with context:
        if "cudf" in str(constructor) or (
            "pyarrow_table" in str(constructor) and PYARROW_VERSION < (14, 0, 0)
        ):
            request.applymarker(pytest.mark.xfail)

        data = {
            "date": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
            "AAPL": [110, 100, 105],
            "TSLA": [220, 200, 210],
            "MSFT": [330, 300, 315],
            "NFLX": [420, 400, 440],
        }

        df = nw.from_native(constructor(data))

        result = df.unpivot(index="date", value_name="price").sort(
            by=["date", "variable"]
        )

        expected = {
            "date": [
                *[date(2020, 1, 1)] * 4,
                *[date(2020, 1, 2)] * 4,
                *[date(2020, 1, 3)] * 4,
            ],
            "variable": [*["AAPL", "MSFT", "NFLX", "TSLA"] * 3],
            "price": [110, 330, 420, 220, 100, 300, 400, 200, 105, 315, 440, 210],
        }
        assert_equal_data(result, expected)
