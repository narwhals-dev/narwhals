from __future__ import annotations

from datetime import datetime

import narwhals as nw
from tests.utils import Constructor, assert_equal_data

data = {
    "flight_date": [
        datetime(2022, 1, 14),
        datetime(2022, 1, 14),
        datetime(2022, 1, 22),
        datetime(2022, 1, 22),
        datetime(2022, 1, 22),
        datetime(2022, 1, 14),
    ],
    "tail_number": ["N119HQ", "N15710", "N312FR", "N410WN", "N420YX", "N467AS"],
    "dep_time": ["1221", "1501", "0950", "0604", "2015", "0909"],
    "dep_delay": [3.0, 1.0, 6.0, 4.0, 25.0, 16.0],
}

expected = {
    "flight_date": [datetime(2022, 1, 22), datetime(2022, 1, 22), datetime(2022, 1, 14)],
    "tail_number": ["N312FR", "N420YX", "N467AS"],
    "dep_time": ["0950", "2015", "0909"],
    "dep_delay": [6.0, 25.0, 16.0],
}


def test_filter_is_between(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = (
        df.select(nw.col(["flight_date", "tail_number", "dep_time", "dep_delay"]))
        .drop_nulls()
        .filter(nw.col("dep_delay").is_between(5, 50, closed="none"))
    )
    assert_equal_data(result, expected)
