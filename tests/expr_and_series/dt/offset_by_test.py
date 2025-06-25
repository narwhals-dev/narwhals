from __future__ import annotations

from datetime import datetime

import pytest

import narwhals as nw
from tests.utils import Constructor, assert_equal_data

data = {
    "a": [datetime(2021, 3, 1, 12, 34, 56, 49012), datetime(2020, 1, 2, 2, 4, 14, 715123)]
}


@pytest.mark.parametrize(
    ("by", "expected"),
    [
        (
            "1us",
            [
                datetime(2021, 3, 1, 12, 34, 56, 49013),
                datetime(2020, 1, 2, 2, 4, 14, 715124),
            ],
        ),
        (
            "1ms",
            [
                datetime(2021, 3, 1, 12, 34, 56, 50012),
                datetime(2020, 1, 2, 2, 4, 14, 716123),
            ],
        ),
        (
            "1s",
            [
                datetime(2021, 3, 1, 12, 34, 57, 49012),
                datetime(2020, 1, 2, 2, 4, 15, 715123),
            ],
        ),
        (
            "1m",
            [
                datetime(2021, 3, 1, 12, 35, 56, 49012),
                datetime(2020, 1, 2, 2, 5, 14, 715123),
            ],
        ),
        (
            "1h",
            [
                datetime(2021, 3, 1, 13, 34, 56, 49012),
                datetime(2020, 1, 2, 3, 4, 14, 715123),
            ],
        ),
        (
            "1d",
            [
                datetime(2021, 3, 2, 12, 34, 56, 49012),
                datetime(2020, 1, 3, 2, 4, 14, 715123),
            ],
        ),
        (
            "1mo",
            [
                datetime(2021, 4, 1, 12, 34, 56, 49012),
                datetime(2020, 2, 2, 2, 4, 14, 715123),
            ],
        ),
        (
            "1q",
            [
                datetime(2021, 6, 1, 12, 34, 56, 49012),
                datetime(2020, 4, 2, 2, 4, 14, 715123),
            ],
        ),
        (
            "1y",
            [
                datetime(2022, 3, 1, 12, 34, 56, 49012),
                datetime(2021, 1, 2, 2, 4, 14, 715123),
            ],
        ),
    ],
)
def test_offset_by(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    by: str,
    expected: list[datetime],
) -> None:
    if any(x in str(constructor) for x in ("ibis", "sqlframe")):
        request.applymarker(pytest.mark.xfail())
    if by.endswith("ns") and any(
        x in str(constructor) for x in ("dask", "duckdb", "pyarrow", "pyspark", "polars")
    ):
        request.applymarker(pytest.mark.xfail())
    if any(x in by for x in ("y", "q", "mo")) and any(
        x in str(constructor) for x in ("dask", "pandas", "pyarrow")
    ):
        request.applymarker(pytest.mark.xfail())
    if by.endswith("d") and any(x in str(constructor) for x in ("dask",)):
        request.applymarker(pytest.mark.xfail())
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").dt.offset_by(by))
    assert_equal_data(result, {"a": expected})
