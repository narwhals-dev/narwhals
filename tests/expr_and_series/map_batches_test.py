from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from narwhals.dependencies import is_dask_dataframe
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {"a": [1, 2, 3], "b": [4, 5, 6], "z": [7.0, 8.0, 9.0]}


def test_map_batches_expr(constructor: Constructor) -> None:
    if is_dask_dataframe(constructor(data)):  # Remove
        pytest.skip()
    df = nw.from_native(constructor(data))
    e = df.select(nw.col("a", "b").map_batches(lambda s: s + 1))
    assert_equal_data(e, {"a": [2, 3, 4], "b": [5, 6, 7]})


def test_map_batches_expr_numpy(constructor: Constructor) -> None:
    if is_dask_dataframe(constructor(data)):  # Remove
        pytest.skip()
    df = nw.from_native(constructor(data))
    e = df.select(
        nw.col("a")
        .map_batches(lambda s: s.to_numpy() + 1, return_dtype=nw.Float64())
        .sum()
    )
    assert_equal_data(e, {"a": [9.0]})
