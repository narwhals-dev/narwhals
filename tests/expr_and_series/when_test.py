from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from narwhals.stable.v1 import when
from tests.utils import compare_dicts

data = {
    "a": [1, 2, 3, 4, 5],
    "b": ["a", "b", "c", "d", "e"],
    "c": [4.1, 5.0, 6.0, 7.0, 8.0],
    "d": [True, False, True, False, True],
}


def test_when(request: Any, constructor: Any) -> None:
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.with_columns(when(nw.col("a") == 1).then(value=3).alias("a_when"))
    expected = {
        "a": [1, 2, 3, 4, 5],
        "b": ["a", "b", "c", "d", "e"],
        "c": [4.1, 5.0, 6.0, 7.0, 8.0],
        "d": [True, False, True, False, True],
        "a_when": [3, None, None, None, None],
    }
    compare_dicts(result, expected)


def test_when_otherwise(request: Any, constructor: Any) -> None:
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.with_columns(when(nw.col("a") == 1).then(3).otherwise(6).alias("a_when"))
    expected = {
        "a": [1, 2, 3, 4, 5],
        "b": ["a", "b", "c", "d", "e"],
        "c": [4.1, 5.0, 6.0, 7.0, 8.0],
        "d": [True, False, True, False, True],
        "a_when": [3, 6, 6, 6, 6],
    }
    compare_dicts(result, expected)


def test_chained_when(request: Any, constructor: Any) -> None:
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.with_columns(
        when(nw.col("a") == 1)
        .then(3)
        .when(nw.col("a") == 2)
        .then(5)
        .otherwise(7)
        .alias("a_when"),
    )
    expected = {
        "a": [1, 2, 3, 4, 5],
        "b": ["a", "b", "c", "d", "e"],
        "c": [4.1, 5.0, 6.0, 7.0, 8.0],
        "d": [True, False, True, False, True],
        "a_when": [3, 5, 7, 7, 7],
    }
    compare_dicts(result, expected)


def test_when_with_multiple_conditions(request: Any, constructor: Any) -> None:
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.with_columns(
        when(nw.col("a") == 1)
        .then(3)
        .when(nw.col("a") == 2)
        .then(5)
        .when(nw.col("a") == 3)
        .then(7)
        .otherwise(9)
        .alias("a_when"),
    )
    expected = {
        "a": [1, 2, 3, 4, 5],
        "b": ["a", "b", "c", "d", "e"],
        "c": [4.1, 5.0, 6.0, 7.0, 8.0],
        "d": [True, False, True, False, True],
        "a_when": [3, 5, 7, 9, 9],
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
        "a": [1, 2, 3, 4, 5],
        "b": ["a", "b", "c", "d", "e"],
        "c": [4.1, 5.0, 6.0, 7.0, 8.0],
        "d": [True, False, True, False, True],
        "a_when": [3, None, None, None, None],
    }
    compare_dicts(result, expected)


def test_when_constraint(request: Any, constructor: Any) -> None:
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.with_columns(when(a=1).then(value=3).alias("a_when"))
    expected = {
        "a": [1, 2, 3, 4, 5],
        "b": ["a", "b", "c", "d", "e"],
        "c": [4.1, 5.0, 6.0, 7.0, 8.0],
        "d": [True, False, True, False, True],
        "a_when": [3, None, None, None, None],
    }
    compare_dicts(result, expected)


def test_no_arg_when_fail(request: Any, constructor: Any) -> None:
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    with pytest.raises(TypeError):
        df.with_columns(when().then(value=3).alias("a_when"))
