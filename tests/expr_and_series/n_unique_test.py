from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {
    "a": [1.0, None, None, 3.0],
    "b": [1.0, None, 4, 5.0],
}


def test_n_unique(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(nw.all().n_unique())
    expected = {"a": [3], "b": [4]}
    assert_equal_data(result, expected)


def test_n_unique_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    expected = {"a": [3], "b": [4]}
    result_series = {"a": [df["a"].n_unique()], "b": [df["b"].n_unique()]}
    assert_equal_data(result_series, expected)
