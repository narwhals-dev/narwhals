from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {
    "a": [
        datetime(2021, 3, 1, 12, 34, 56, 49000),
        datetime(2020, 1, 2, 2, 4, 14, 715000),
    ],
}


@pytest.mark.parametrize(
    ("attribute", "expected"),
    [
        ("year", [2021, 2020]),
        ("month", [3, 1]),
        ("day", [1, 2]),
        ("hour", [12, 2]),
        ("minute", [34, 4]),
        ("second", [56, 14]),
        ("millisecond", [49, 715]),
        ("microsecond", [49000, 715000]),
        ("nanosecond", [49000000, 715000000]),
        ("ordinal_day", [60, 2]),
    ],
)
def test_datetime_attributes(
    constructor: Any, attribute: str, expected: list[int]
) -> None:
    df = nw.from_native(constructor(data), eager_only=True)
    result = df.select(getattr(nw.col("a").dt, attribute)())
    compare_dicts(result, {"a": expected})

    result = df.select(getattr(df["a"].dt, attribute)())
    compare_dicts(result, {"a": expected})
