from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data


@pytest.mark.parametrize("n", [2, -1])
def test_head_series(nw_eager_constructor: ConstructorEager, n: int) -> None:
    df = nw.from_native(nw_eager_constructor({"a": [1, 2, 3]}), eager_only=True)
    result = df.select(df["a"].head(n))
    expected = {"a": [1, 2]}
    assert_equal_data(result, expected)
