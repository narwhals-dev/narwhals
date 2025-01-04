from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


@pytest.mark.parametrize("n", [2, -1])
def test_head(
    constructor_eager: ConstructorEager, n: int, request: pytest.FixtureRequest
) -> None:
    if "polars" in str(constructor_eager) and n < 0:
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}))
    result = df.select(nw.col("a").head(n))
    expected = {"a": [1, 2]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize("n", [2, -1])
def test_head_series(constructor_eager: ConstructorEager, n: int) -> None:
    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)
    result = df.select(df["a"].head(n))
    expected = {"a": [1, 2]}
    assert_equal_data(result, expected)
