from __future__ import annotations

import re

import pytest
from tests.utils import ConstructorEager, assert_equal_data

import narwhals as nw

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


def test_out_of_range(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager({"a": [1, 2]}), eager_only=True)["a"]
    with pytest.raises(IndexError, match="out of range"):
        series.item(2)
