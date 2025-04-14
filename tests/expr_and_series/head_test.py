from __future__ import annotations

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


@pytest.mark.parametrize("n", [2, -1])
def test_head(
    constructor_eager: ConstructorEager, n: int, request: pytest.FixtureRequest
) -> None:
    if "polars" in str(constructor_eager) and n < 0:
        request.applymarker(pytest.mark.xfail)
    df = nw_v1.from_native(constructor_eager({"a": [1, 2, 3]}))
    result = df.select(nw_v1.col("a").head(n))
    expected = {"a": [1, 2]}
    assert_equal_data(result, expected)

    with pytest.deprecated_call(
        match="is deprecated and will be removed in a future version"
    ):
        df.select(nw.col("a").head(5))


@pytest.mark.parametrize("n", [2, -1])
def test_head_series(constructor_eager: ConstructorEager, n: int) -> None:
    df = nw_v1.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)
    result = df.select(df["a"].head(n))
    expected = {"a": [1, 2]}
    assert_equal_data(result, expected)
