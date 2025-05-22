from __future__ import annotations

import re

import pytest

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data

data = [1, 3, 2]


@pytest.mark.parametrize(("index", "expected"), [(0, 1), (1, 3)])
def test_item(constructor_eager: ConstructorEager, index: int, expected: int) -> None:
    series = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]
    result = series.item(index)
    assert_equal_data({"a": [result]}, {"a": [expected]})
    assert_equal_data({"a": [series.head(1).item()]}, {"a": [1]})

    with pytest.raises(
        ValueError,
        match=re.escape("can only call '.item()' if the Series is of length 1,"),
    ):
        series.item(None)
