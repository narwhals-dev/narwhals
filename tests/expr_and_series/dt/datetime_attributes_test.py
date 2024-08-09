from __future__ import annotations

from datetime import date
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
        ("date", [date(2021, 3, 1), date(2020, 1, 2)]),
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
    request: Any, constructor: Any, attribute: str, expected: list[int]
) -> None:
    if (
        attribute == "date"
        and "pandas" in str(constructor)
        and "pyarrow" not in str(constructor)
    ):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select(getattr(nw.col("a").dt, attribute)())
    compare_dicts(result, {"a": expected})


@pytest.mark.parametrize(
    ("attribute", "expected"),
    [
        ("date", [date(2021, 3, 1), date(2020, 1, 2)]),
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
def test_datetime_attributes_series(
    request: Any, constructor_eager: Any, attribute: str, expected: list[int]
) -> None:
    if (
        attribute == "date"
        and "pandas" in str(constructor_eager)
        and "pyarrow" not in str(constructor_eager)
    ):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(getattr(df["a"].dt, attribute)())
    compare_dicts(result, {"a": expected})


def test_datetime_chained_attributes(request: Any, constructor_eager: Any) -> None:
    if "pandas" in str(constructor_eager) and "pyarrow" not in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(df["a"].dt.date().dt.year())
    compare_dicts(result, {"a": [2021, 2020]})

    result = df.select(nw.col("a").dt.date().dt.year())
    compare_dicts(result, {"a": [2021, 2020]})
