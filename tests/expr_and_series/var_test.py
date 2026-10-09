from __future__ import annotations

import numpy as np
import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
data_with_nulls = {"a": [1, 3, 2, None], "b": [4, 4, 6, None], "z": [7.0, 8.0, 9.0, None]}

expected_results = {
    "a_ddof_1": [1.0],
    "a_ddof_0": [0.6666666666666666],
    "b_ddof_2": [2.666666666666667],
    "z_ddof_0": [0.6666666666666666],
}


@pytest.mark.parametrize("input_data", [data, data_with_nulls])
def test_var(constructor: Constructor, input_data: dict[str, list[float | None]]) -> None:
    df = nw.from_native(constructor(input_data))
    result = df.select(
        nw.col("a").var(ddof=1).alias("a_ddof_1"),
        nw.col("a").var(ddof=0).alias("a_ddof_0"),
        nw.col("z").var(ddof=0).alias("z_ddof_0"),
    )
    expected_results = {
        "a_ddof_1": [1.0],
        "a_ddof_0": [0.6666666666666666],
        "z_ddof_0": [0.6666666666666666],
    }
    assert_equal_data(result, expected_results)

    result = df.select(nw.col("b").var(ddof=2).alias("b_ddof_2"))
    expected_results = {"b_ddof_2": [2.666666666666667]}
    assert_equal_data(result, expected_results)


@pytest.mark.parametrize("input_data", [data, data_with_nulls])
def test_var_series(
    constructor_eager: ConstructorEager, input_data: dict[str, list[float | None]]
) -> None:
    df = nw.from_native(constructor_eager(input_data), eager_only=True)
    result = {
        "a_ddof_1": [df["a"].var(ddof=1)],
        "a_ddof_0": [df["a"].var(ddof=0)],
        "b_ddof_2": [df["b"].var(ddof=2)],
        "z_ddof_0": [df["z"].var(ddof=0)],
    }
    assert_equal_data(result, expected_results)


data_ddof: dict[str, list[float]] = {
    "i": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "g": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    "a": [1.0, 2.0, 2.0, 5.0, -7.0, 4.0, 3.0, 0.5, 8.0, -1.0, 6.0],
}


@pytest.mark.parametrize("ddof", [0, 1, 2, 4])
def test_var_ddof(constructor: Constructor, ddof: int) -> None:
    df = nw.from_native(constructor(data_ddof))
    result = df.select(nw.col("a").var(ddof=ddof))
    expected = {"a": [float(np.var(data_ddof["a"], ddof=ddof))]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize("ddof", [0, 1, 2, 4])
def test_var_ddof_group_by(constructor: Constructor, ddof: int) -> None:
    df = nw.from_native(constructor(data_ddof))
    result = df.group_by("g").agg(nw.col("a").var(ddof=ddof)).sort("g")
    expected = {
        "g": [1, 2],
        "a": [
            float(np.var(data_ddof["a"][:6], ddof=ddof)),
            float(np.var(data_ddof["a"][6:], ddof=ddof)),
        ],
    }
    assert_equal_data(result, expected)


@pytest.mark.parametrize("ddof", [0, 1, 2, 4])
def test_var_ddof_over(
    constructor: Constructor, ddof: int, request: pytest.FixtureRequest
) -> None:
    if "duckdb" in str(constructor) and ddof > 1:
        # The rescaled expression is wrapped as a single window function, see #4023.
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data_ddof))
    result = df.with_columns(nw.col("a").var(ddof=ddof).over("g")).sort("i").select("a")
    expected = {
        "a": [float(np.var(data_ddof["a"][:6], ddof=ddof))] * 6
        + [float(np.var(data_ddof["a"][6:], ddof=ddof))] * 5
    }
    assert_equal_data(result, expected)
