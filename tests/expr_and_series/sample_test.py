import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor


def test_expr_sample(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]})).lazy()

    result_expr = df.select(nw.col("a").sample(n=2)).collect().shape
    expected_expr = (2, 1)
    assert result_expr == expected_expr

    result_series = df.collect()["a"].sample(n=2).shape
    expected_series = (2,)
    assert result_series == expected_series


def test_expr_sample_fraction(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor({"a": [1, 2, 3] * 10, "b": [4, 5, 6] * 10})).lazy()

    result_expr = df.select(nw.col("a").sample(fraction=0.1)).collect().shape
    expected_expr = (3, 1)
    assert result_expr == expected_expr

    result_series = df.collect()["a"].sample(fraction=0.1).shape
    expected_series = (3,)
    assert result_series == expected_series
