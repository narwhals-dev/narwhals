from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import compare_dicts

data = {
    "a": [1, 2, 3],
    "b": ["a", "b", "c"],
    "c": [4.1, 5.0, 6.0],
    "d": [True, False, True],
    "e": [7.0, 2.0, 1.1],
}

large_data = {
    "a": [1, 2, 3, 4, 5, 6],
    "b": ["a", "b", "c", "d", "e", "f"],
    "c": [True, False, True, False, True, False],
}


def test_when(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.when(nw.col("a") == 1).then(value=3).alias("a_when"))
    expected = {
        "a_when": [3, np.nan, np.nan],
    }
    compare_dicts(result, expected)


def test_when_otherwise(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.when(nw.col("a") == 1).then(3).otherwise(6).alias("a_when"))
    expected = {
        "a_when": [3, 6, 6],
    }
    compare_dicts(result, expected)


def test_multiple_conditions(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(
        nw.when(nw.col("a") < 3, nw.col("c") < 5.0).then(3).alias("a_when")
    )
    expected = {
        "a_when": [3, np.nan, np.nan],
    }
    compare_dicts(result, expected)


def test_no_arg_when_fail(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    with pytest.raises((TypeError, ValueError)):
        df.select(nw.when().then(value=3).alias("a_when"))


def test_value_numpy_array(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
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


def test_value_expression(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.when(nw.col("a") == 1).then(nw.col("a") + 9).alias("a_when"))
    expected = {
        "a_when": [10, np.nan, np.nan],
    }
    compare_dicts(result, expected)


def test_otherwise_numpy_array(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
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


def test_otherwise_expression(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
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


def test_when_then_otherwise_into_expr(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select(nw.when(nw.col("a") > 1).then("c").otherwise("e").alias("a_when"))
    expected = {"a_when": [7, 5, 6]}
    compare_dicts(result, expected)


def test_chained_when(request: Any, constructor: Any) -> None:
    if "dask" in str(constructor) or "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select(
        nw.when(nw.col("a") == 1).then(3).when(nw.col("a") == 2).then(5).alias("a_when"),
    )
    expected = {
        "a_when": [3, 5, np.nan],
    }
    compare_dicts(result, expected)


def test_chained_when_otherewise(request: Any, constructor: Any) -> None:
    if "dask" in str(constructor) or "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select(
        nw.when(nw.col("a") == 1)
        .then(3)
        .when(nw.col("a") == 2)
        .then(5)
        .otherwise(7)
        .alias("a_when"),
    )
    expected = {
        "a_when": [3, 5, 7],
    }
    compare_dicts(result, expected)


def test_multi_chained_when(request: Any, constructor: Any) -> None:
    if "dask" in str(constructor) or "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(large_data))
    result = df.select(
        nw.when(nw.col("a") == 1)
        .then(3)
        .when(nw.col("a") == 2)
        .then(5)
        .when(nw.col("a") == 3)
        .then(7)
        .alias("a_when"),
    )
    expected = {
        "a_when": [3, 5, 7, np.nan, np.nan, np.nan],
    }
    compare_dicts(result, expected)


def test_multi_chained_when_otherwise(request: Any, constructor: Any) -> None:
    if "dask" in str(constructor) or "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(large_data))
    result = df.select(
        nw.when(nw.col("a") == 1)
        .then(3)
        .when(nw.col("a") == 2)
        .then(5)
        .when(nw.col("a") == 3)
        .then(7)
        .otherwise(9)
        .alias("a_when"),
    )
    expected = {
        "a_when": [3, 5, 7, 9, 9, 9],
    }
    compare_dicts(result, expected)


def test_then_when_no_condition(request: Any, constructor: Any) -> None:
    if "dask" in str(constructor) or "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))

    with pytest.raises((TypeError, ValueError)):
        df.select(nw.when(nw.col("a") == 1).then(value=3).when().then(value=7))


def test_then_chained_when_no_condition(request: Any, constructor: Any) -> None:
    if "dask" in str(constructor) or "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))

    with pytest.raises((TypeError, ValueError)):
        df.select(
            nw.when(nw.col("a") == 1)
            .then(value=3)
            .when(nw.col("a") == 3)
            .then(value=7)
            .when()
            .then(value=9)
        )
