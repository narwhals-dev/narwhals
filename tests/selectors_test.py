from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from narwhals.stable.v1.selectors import all
from narwhals.stable.v1.selectors import boolean
from narwhals.stable.v1.selectors import by_dtype
from narwhals.stable.v1.selectors import categorical
from narwhals.stable.v1.selectors import numeric
from narwhals.stable.v1.selectors import string
from tests.utils import PYARROW_VERSION
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {
    "a": [1, 1, 2],
    "b": ["a", "b", "c"],
    "c": [4.1, 5.0, 6.0],
    "d": [True, False, True],
}


def test_selectors(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if ("pyspark" in str(constructor)) or "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(by_dtype([nw.Int64, nw.Float64]) + 1)
    expected = {"a": [2, 2, 3], "c": [5.1, 6.0, 7.0]}
    assert_equal_data(result, expected)


def test_numeric(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if ("pyspark" in str(constructor)) or "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(numeric() + 1)
    expected = {"a": [2, 2, 3], "c": [5.1, 6.0, 7.0]}
    assert_equal_data(result, expected)


def test_boolean(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if ("pyspark" in str(constructor)) or "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(boolean())
    expected = {"d": [True, False, True]}
    assert_equal_data(result, expected)


def test_string(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if ("pyspark" in str(constructor)) or "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(string())
    expected = {"b": ["a", "b", "c"]}
    assert_equal_data(result, expected)


def test_categorical(
    request: pytest.FixtureRequest,
    constructor: Constructor,
) -> None:
    if "pyarrow_table_constructor" in str(constructor) and PYARROW_VERSION <= (
        15,
    ):  # pragma: no cover
        request.applymarker(pytest.mark.xfail)
    if ("pyspark" in str(constructor)) or "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    expected = {"b": ["a", "b", "c"]}

    df = nw.from_native(constructor(data)).with_columns(nw.col("b").cast(nw.Categorical))
    result = df.select(categorical())
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        (numeric() | boolean(), ["a", "c", "d"]),
        (numeric() & boolean(), []),
        (numeric() & by_dtype(nw.Int64), ["a"]),
        (numeric() | by_dtype(nw.Int64), ["a", "c"]),
        (~numeric(), ["b", "d"]),
        (boolean() & True, ["d"]),
        (boolean() | True, ["d"]),
        (numeric() - 1, ["a", "c"]),
        (all(), ["a", "b", "c", "d"]),
    ],
)
def test_set_ops(
    constructor: Constructor,
    selector: nw.selectors.Selector,
    expected: list[str],
    request: pytest.FixtureRequest,
) -> None:
    if ("pyspark" in str(constructor)) or "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(selector).collect_schema().names()
    assert sorted(result) == expected


@pytest.mark.parametrize("invalid_constructor", [pd.DataFrame, pa.table])
def test_set_ops_invalid(
    invalid_constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "duckdb" in str(invalid_constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(invalid_constructor(data))
    with pytest.raises((NotImplementedError, ValueError)):
        df.select(1 - numeric())
    with pytest.raises((NotImplementedError, ValueError)):
        df.select(1 | numeric())
    with pytest.raises((NotImplementedError, ValueError)):
        df.select(1 & numeric())
