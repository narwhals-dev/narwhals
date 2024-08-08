from __future__ import annotations

from typing import Any

import pytest

import narwhals as nw
from tests.utils import compare_dicts

data = {
    "a": [1, 2, None],
    "b": [3, 4, 5],
    "c": [None, None, None],
    "d": [6, None, None],
}


def test_drop_nulls(constructor: Any) -> None:
    df = nw.from_native(constructor(data))

    result_a = df.select(nw.col("a").drop_nulls())
    result_b = df.select(nw.col("b").drop_nulls())
    result_c = df.select(nw.col("c").drop_nulls())
    result_d = df.select(nw.col("d").drop_nulls())

    expected_a = {"a": [1.0, 2.0]}
    expected_b = {"b": [3, 4, 5]}
    expected_c = {"c": []}  # type: ignore[var-annotated]
    expected_d = {"d": [6]}

    compare_dicts(result_a, expected_a)
    compare_dicts(result_b, expected_b)
    compare_dicts(result_c, expected_c)
    compare_dicts(result_d, expected_d)


def test_drop_nulls_broadcast(constructor: Any, request: Any) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").drop_nulls(), nw.col("d").drop_nulls())
    expected = {"a": [1.0, 2.0], "d": [6, 6]}
    compare_dicts(result, expected)


def test_drop_nulls_invalid(constructor: Any) -> None:
    df = nw.from_native(constructor(data)).lazy()

    with pytest.raises(Exception):  # noqa: B017, PT011
        df.select(nw.col("a").drop_nulls(), nw.col("b").drop_nulls()).collect()


def test_drop_nulls_series(constructor_eager: Any) -> None:
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

    compare_dicts(result_a, expected_a)
    compare_dicts(result_b, expected_b)
    compare_dicts(result_c, expected_c)
    compare_dicts(result_d, expected_d)
