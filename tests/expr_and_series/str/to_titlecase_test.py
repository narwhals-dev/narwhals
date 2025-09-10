from __future__ import annotations

from copy import deepcopy

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {
    "a": [
        "e.t. phone home",
        "they're bill's friends from the UK",
        "to infinity,and BEYOND!",
        "with123numbers",
    ]
}
expected = {
    "a": [
        "E.T. Phone Home",
        "They'Re Bill'S Friends From The Uk",
        "To Infinity,And Beyond!",
        "With123Numbers",
    ]
}


def test_str_to_titlecase(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(x in str(constructor) for x in ("ibis", "pyspark", "sqlframe")):
        request.applymarker(pytest.mark.xfail)

    expected_ = deepcopy(expected)
    if "polars" in str(constructor) or "duckdb" in str(constructor):
        expected_["a"][-1] = "With123numbers"

    df = nw.from_native(constructor(data))
    result_frame = df.select(nw.col("a").str.to_titlecase())

    assert_equal_data(result_frame, expected_)


def test_str_to_titlecase_series(constructor_eager: ConstructorEager) -> None:
    expected_ = deepcopy(expected)
    if "polars" in str(constructor_eager):
        expected_["a"][-1] = "With123numbers"

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result_series = df["a"].str.to_titlecase()

    assert_equal_data({"a": result_series}, expected_)
