from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
data_with_nulls = {"a": [1, 3, 2, None], "b": [4, 4, 6, None], "z": [7.0, 8, 9, None]}

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
    context = (
        pytest.raises(NotImplementedError)
        if "duckdb" in str(constructor)
        else does_not_raise()
    )
    with context:
        result = df.select(
            nw.col("b").var(ddof=2).alias("b_ddof_2"),
        )
        expected_results = {
            "b_ddof_2": [2.666666666666667],
        }
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
