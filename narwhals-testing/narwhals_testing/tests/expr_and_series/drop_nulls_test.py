from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data


def test_drop_nulls(constructor_eager: ConstructorEager) -> None:
    data = {
        "A": [1, 2, None, 4],
        "B": [5, 6, 7, 8],
        "C": [None, None, None, None],
        "D": [9, 10, 11, 12],
    }

    df = nw.from_native(constructor_eager(data))

    result_a = df.select(nw.col("A").drop_nulls())
    result_b = df.select(nw.col("B").drop_nulls())
    result_c = df.select(nw.col("C").drop_nulls())
    result_d = df.select(nw.col("D").drop_nulls())
    expected_a = {"A": [1.0, 2.0, 4.0]}
    expected_b = {"B": [5, 6, 7, 8]}
    expected_c = {"C": []}  # type: ignore[var-annotated]
    expected_d = {"D": [9, 10, 11, 12]}

    assert_equal_data(result_a, expected_a)
    assert_equal_data(result_b, expected_b)
    assert_equal_data(result_c, expected_c)
    assert_equal_data(result_d, expected_d)


def test_drop_nulls_agg(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if any(x in str(constructor) for x in ("duckdb", "pyspark", "ibis")):
        request.applymarker(pytest.mark.xfail)
    data = {
        "A": [1, 2, None, 4],
        "B": [5, 6, 7, 8],
        "C": [None, None, None, None],
        "D": [9, 10, 11, 12],
    }

    df = nw.from_native(constructor(data))
    result = df.select(nw.all().drop_nulls().len())
    expected = {"A": [3], "B": [4], "C": [0], "D": [4]}
    assert_equal_data(result, expected)


def test_drop_nulls_series(constructor_eager: ConstructorEager) -> None:
    data = {
        "A": [1, 2, None, 4],
        "B": [5, 6, 7, 8],
        "C": [None, None, None, None],
        "D": [9, 10, 11, 12],
    }

    df = nw.from_native(constructor_eager(data), eager_only=True)

    result_a = df.select(df["A"].drop_nulls())
    result_b = df.select(df["B"].drop_nulls())
    result_c = df.select(df["C"].drop_nulls())
    result_d = df.select(df["D"].drop_nulls())
    expected_a = {"A": [1.0, 2.0, 4.0]}
    expected_b = {"B": [5, 6, 7, 8]}
    expected_c = {"C": []}  # type: ignore[var-annotated]
    expected_d = {"D": [9, 10, 11, 12]}

    assert_equal_data(result_a, expected_a)
    assert_equal_data(result_b, expected_b)
    assert_equal_data(result_c, expected_c)
    assert_equal_data(result_d, expected_d)
