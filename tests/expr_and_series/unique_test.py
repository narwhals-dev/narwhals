from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {"a": [1, 1, 2]}


def test_unique_expr(constructor: Any, request: pytest.FixtureRequest) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").unique())
    expected = {"a": [1, 2]}
    compare_dicts(result, expected)


def test_unique_series(constructor_eager: Any) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]
    result = series.unique()
    expected = {"a": [1, 2]}
    compare_dicts({"a": result}, expected)
