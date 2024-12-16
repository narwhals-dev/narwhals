from __future__ import annotations

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}


def test_var(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(
        nw.col("a").var().alias("a_ddof_default"),
        nw.col("a").var(ddof=1).alias("a_ddof_1"),
        nw.col("a").var(ddof=0).alias("a_ddof_0"),
        nw.col("b").var(ddof=2).alias("b_ddof_2"),
        nw.col("z").var(ddof=0).alias("z_ddof_0"),
    )
    expected = {
        "a_ddof_default": [1.0],
        "a_ddof_1": [1.0],
        "a_ddof_0": [0.6666666666666666],
        "b_ddof_2": [2.666666666666667],
        "z_ddof_0": [0.6666666666666666],
    }
    assert_equal_data(result, expected)


def test_var_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = {
        "a_ddof_default": [df["a"].var()],
        "a_ddof_1": [df["a"].var(ddof=1)],
        "a_ddof_0": [df["a"].var(ddof=0)],
        "b_ddof_2": [df["b"].var(ddof=2)],
        "z_ddof_0": [df["z"].var(ddof=0)],
    }
    expected = {
        "a_ddof_default": [1.0],
        "a_ddof_1": [1.0],
        "a_ddof_0": [0.6666666666666666],
        "b_ddof_2": [2.666666666666667],
        "z_ddof_0": [0.6666666666666666],
    }
    assert_equal_data(result, expected)
