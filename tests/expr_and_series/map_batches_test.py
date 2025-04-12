from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import POLARS_VERSION
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [1, 2, 3], "b": [4, 5, 6], "z": [7.0, 8.0, 9.0]}


def test_map_batches_expr(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    expected = df.select(nw.col("a", "b").map_batches(lambda s: s + 1))
    assert_equal_data(expected, {"a": [2, 3, 4], "b": [5, 6, 7]})


def test_map_batches_expr_numpy(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "polars" in str(constructor_eager) and POLARS_VERSION <= (0, 20, 3):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data))
    expected = df.select(
        nw.col("a")
        .map_batches(lambda s: s.to_numpy() + 1, return_dtype=nw.Float64())
        .sum()
    )
    assert_equal_data(expected, {"a": [9.0]})

    expected = df.select(nw.all().map_batches(lambda s: s.to_numpy().argmax()))
    assert_equal_data(expected, {"a": [2], "b": [2], "z": [2]})


def test_map_batches_expr_names(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "polars" in str(constructor_eager) and POLARS_VERSION <= (0, 20, 3):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data))
    expected = nw.from_native(df.select(nw.all().map_batches(lambda x: x.to_numpy())))
    assert_equal_data(expected, {"a": [1, 2, 3], "b": [4, 5, 6], "z": [7.0, 8.0, 9.0]})
