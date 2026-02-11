from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data


@pytest.mark.parametrize("n", [2, -1])
def test_tail_series(constructor_eager: ConstructorEager, n: int) -> None:
    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)
    result = df.select(df["a"].tail(n))
    expected = {"a": [2, 3]}
    assert_equal_data(result, expected)
