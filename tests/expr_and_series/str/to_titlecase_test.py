from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": ["e.t. phone home", "you talkin' to me?", "to infinity,and BEYOND!"]}
expected = {"a": ["E.T. Phone Home", "You Talkin' To Me?", "To Infinity,And Beyond!"]}


def test_str_to_titlecase(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if any(x in str(constructor) for x in ("duckdb", "ibis", "pyspark", "sqlframe")):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result_frame = df.select(nw.col("a").str.to_titlecase())

    assert_equal_data(result_frame, expected)


def test_str_to_titlecase_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result_series = df["a"].str.to_titlecase()
    assert_equal_data({"a": result_series}, expected)
