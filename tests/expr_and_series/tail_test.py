from __future__ import annotations

import pytest

import narwhals as nw_main
import narwhals.stable.v1 as nw
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


@pytest.mark.parametrize("n", [2, -1])
def test_tail(
    constructor_eager: ConstructorEager, n: int, request: pytest.FixtureRequest
) -> None:
    if "polars" in str(constructor_eager) and n < 0:
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}))
    result = df.select(nw.col("a").tail(n))
    expected = {"a": [2, 3]}
    assert_equal_data(result, expected)

    with pytest.deprecated_call():
        df.select(nw_main.col("a").tail(5))


@pytest.mark.parametrize("n", [2, -1])
def test_tail_series(constructor_eager: ConstructorEager, n: int) -> None:
    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)
    result = df.select(df["a"].tail(n))
    expected = {"a": [2, 3]}
    assert_equal_data(result, expected)
