from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {
    "a": [1, 2, 3],
    "b": ["a", "b", "c"],
    "c": [4.1, 5.0, 6.0],
    "d": [True, False, True],
    "e": [7.0, 2.0, 1.1],
}


def test_when(constructor: Any, request: pytest.FixtureRequest) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(nw.when(nw.col("a") == 1).then(value=3).alias("a_when"))
    expected = {
        "a_when": [3, np.nan, np.nan],
    }
    compare_dicts(result, expected)


def test_when_otherwise(constructor: Any, request: pytest.FixtureRequest) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(nw.when(nw.col("a") == 1).then(3).otherwise(6).alias("a_when"))
    expected = {
        "a_when": [3, 6, 6],
    }
    compare_dicts(result, expected)


def test_multiple_conditions(constructor: Any, request: pytest.FixtureRequest) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(
        nw.when(nw.col("a") < 3, nw.col("c") < 5.0).then(3).alias("a_when")
    )
    expected = {
        "a_when": [3, np.nan, np.nan],
    }
    compare_dicts(result, expected)


def test_no_arg_when_fail(constructor: Any, request: pytest.FixtureRequest) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    with pytest.raises((TypeError, ValueError)):
        df.select(nw.when().then(value=3).alias("a_when"))


def test_value_numpy_array(constructor: Any, request: pytest.FixtureRequest) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    import numpy as np

    result = df.select(
        nw.when(nw.col("a") == 1).then(np.asanyarray([3, 4, 5])).alias("a_when")
    )
    expected = {
        "a_when": [3, np.nan, np.nan],
    }
    compare_dicts(result, expected)


def test_value_series(constructor_eager: Any) -> None:
    df = nw.from_native(constructor_eager(data))
    s_data = {"s": [3, 4, 5]}
    s = nw.from_native(constructor_eager(s_data))["s"]
    assert isinstance(s, nw.Series)
    result = df.select(nw.when(nw.col("a") == 1).then(s).alias("a_when"))
    expected = {
        "a_when": [3, np.nan, np.nan],
    }
    compare_dicts(result, expected)


def test_value_expression(constructor: Any, request: pytest.FixtureRequest) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(nw.when(nw.col("a") == 1).then(nw.col("a") + 9).alias("a_when"))
    expected = {
        "a_when": [10, np.nan, np.nan],
    }
    compare_dicts(result, expected)


def test_otherwise_numpy_array(constructor: Any, request: pytest.FixtureRequest) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    import numpy as np

    result = df.select(
        nw.when(nw.col("a") == 1).then(-1).otherwise(np.array([0, 9, 10])).alias("a_when")
    )
    expected = {
        "a_when": [-1, 9, 10],
    }
    compare_dicts(result, expected)


def test_otherwise_series(constructor_eager: Any) -> None:
    df = nw.from_native(constructor_eager(data))
    s_data = {"s": [0, 9, 10]}
    s = nw.from_native(constructor_eager(s_data))["s"]
    assert isinstance(s, nw.Series)
    result = df.select(nw.when(nw.col("a") == 1).then(-1).otherwise(s).alias("a_when"))
    expected = {
        "a_when": [-1, 9, 10],
    }
    compare_dicts(result, expected)


def test_otherwise_expression(constructor: Any, request: pytest.FixtureRequest) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select(
        nw.when(nw.col("a") == 1).then(-1).otherwise(nw.col("a") + 7).alias("a_when")
    )
    expected = {
        "a_when": [-1, 9, 10],
    }
    compare_dicts(result, expected)


def test_when_then_otherwise_into_expr(constructor: Any, request: pytest.FixtureRequest) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select(nw.when(nw.col("a") > 1).then("c").otherwise("e"))
    expected = {"c": [7, 5, 6]}
    compare_dicts(result, expected)
