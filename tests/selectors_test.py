from __future__ import annotations

import re

import pytest

import narwhals.stable.v1 as nw
from narwhals.stable.v1.selectors import all
from narwhals.stable.v1.selectors import boolean
from narwhals.stable.v1.selectors import by_dtype
from narwhals.stable.v1.selectors import categorical
from narwhals.stable.v1.selectors import numeric
from narwhals.stable.v1.selectors import string
from tests.utils import POLARS_VERSION
from tests.utils import PYARROW_VERSION
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {
    "a": [1, 1, 2],
    "b": ["a", "b", "c"],
    "c": [4.1, 5.0, 6.0],
    "d": [True, False, True],
}


def test_selectors(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(by_dtype([nw.Int64, nw.Float64]) + 1)
    expected = {"a": [2, 2, 3], "c": [5.1, 6.0, 7.0]}
    assert_equal_data(result, expected)


def test_numeric(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(numeric() + 1)
    expected = {"a": [2, 2, 3], "c": [5.1, 6.0, 7.0]}
    assert_equal_data(result, expected)


def test_boolean(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(boolean())
    expected = {"d": [True, False, True]}
    assert_equal_data(result, expected)


def test_string(constructor: Constructor) -> None:
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
    if "pyspark" in str(constructor) or "duckdb" in str(constructor):
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
    if "duckdb" in str(constructor) and not expected:
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(selector).collect_schema().names()
    assert sorted(result) == expected


def test_subtract_expr(
    constructor: Constructor,
    request: pytest.FixtureRequest,
) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (0, 20, 27):
        # In old Polars versions, cs.numeric() - col('a')
        # would exclude column 'a' from the result, as opposed to
        # subtracting it.
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(numeric() - nw.col("a"))
    expected = {"a": [0, 0, 0], "c": [3.1, 4.0, 4.0]}
    assert_equal_data(result, expected)


def test_set_ops_invalid(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    with pytest.raises((NotImplementedError, ValueError)):
        df.select(1 - numeric())
    with pytest.raises((NotImplementedError, ValueError)):
        df.select(1 | numeric())
    with pytest.raises((NotImplementedError, ValueError)):
        df.select(1 & numeric())

    with pytest.raises(
        TypeError,
        match=re.escape("unsupported operand type(s) for op: ('Selector' + 'Selector')"),
    ):
        df.select(boolean() + numeric())
