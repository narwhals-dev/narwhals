from __future__ import annotations

from datetime import datetime

import narwhals as nw
from tests.utils import Constructor, assert_equal_data


def test_filter(constructor: Constructor) -> None:
    data = {
        "time": [
            "2021-01-01",
            "2021-01-02",
            "2021-01-03",
            "2021-01-04",
            "2021-01-05",
            "2021-01-06",
            "2021-01-07",
            "2021-01-08",
            "2021-01-09",
            "2021-01-10",
        ],
        "close": [
            29331.69,
            32178.33,
            33000.05,
            31988.71,
            33949.53,
            36769.36,
            39432.28,
            40582.81,
            40088.22,
            38150.02,
        ],
    }

    df = nw.from_native(constructor(data))

    df = df.with_columns(nw.col("time").str.to_datetime(format="%Y-%m-%d"))

    result = df.filter(
        nw.col("time").is_between(
            datetime(2021, 1, 4), datetime(2021, 1, 7), closed="left"
        )
    )
    expected = {
        "time": [datetime(2021, 1, 4), datetime(2021, 1, 5), datetime(2021, 1, 6)],
        "close": [31988.71, 33949.53, 36769.36],
    }
    assert_equal_data(result, expected)
