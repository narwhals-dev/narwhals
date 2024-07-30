from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from narwhals.stable.v1 import when
from tests.utils import compare_dicts

data = {
    "a": [1, 2, 3],
    "b": ["a", "b", "c"],
    "c": [4.1, 5.0, 6.0],
    "d": [True, False, True],
}


def test_when(request: Any, constructor: Any) -> None:
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.with_columns(when(nw.col("a") == 1).then(value=3).alias("a_when"))
    expected = {
        "a": [1, 2, 3],
        "b": ["a", "b", "c"],
        "c": [4.1, 5.0, 6.0],
        "d": [True, False, True],
        "a_when": [3, None, None],
    }
    compare_dicts(result, expected)


def test_when_otherwise(request: Any, constructor: Any) -> None:
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.with_columns(when(nw.col("a") == 1).then(3).otherwise(6).alias("a_when"))
    expected = {
        "a": [1, 2, 3],
        "b": ["a", "b", "c"],
        "c": [4.1, 5.0, 6.0],
        "d": [True, False, True],
        "a_when": [3, 6, 6],
    }
    compare_dicts(result, expected)


def test_multiple_conditions(request: Any, constructor: Any) -> None:
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.with_columns(
        when(nw.col("a") < 3, nw.col("c") < 5.0).then(3).alias("a_when")
    )
    expected = {
        "a": [1, 2, 3],
        "b": ["a", "b", "c"],
        "c": [4.1, 5.0, 6.0],
        "d": [True, False, True],
        "a_when": [3, None, None],
    }
    compare_dicts(result, expected)


def test_no_arg_when_fail(request: Any, constructor: Any) -> None:
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    with pytest.raises(TypeError):
        df.with_columns(when().then(value=3).alias("a_when"))


def test_value_numpy_array(request: Any, constructor: Any) -> None:
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    import numpy as np

    result = df.with_columns(
        when(nw.col("a") == 1).then(np.asanyarray([3, 4, 5])).alias("a_when")
    )
    expected = {
        "a": [1, 2, 3],
        "b": ["a", "b", "c"],
        "c": [4.1, 5.0, 6.0],
        "d": [True, False, True],
        "a_when": [3, None, None],
    }
    compare_dicts(result, expected)


def test_value_series(request: Any, constructor: Any) -> None:
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    s_data = {"s": [3, 4, 5]}
    s = nw.from_native(constructor(s_data))["s"]
    assert isinstance(s, nw.Series)
    result = df.with_columns(when(nw.col("a") == 1).then(s).alias("a_when"))
    expected = {
        "a": [1, 2, 3],
        "b": ["a", "b", "c"],
        "c": [4.1, 5.0, 6.0],
        "d": [True, False, True],
        "a_when": [3, None, None],
    }
    compare_dicts(result, expected)


def test_value_expression(request: Any, constructor: Any) -> None:
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.with_columns(when(nw.col("a") == 1).then(nw.col("a") + 9).alias("a_when"))
    expected = {
        "a": [1, 2, 3],
        "b": ["a", "b", "c"],
        "c": [4.1, 5.0, 6.0],
        "d": [True, False, True],
        "a_when": [10, None, None],
    }
    compare_dicts(result, expected)
