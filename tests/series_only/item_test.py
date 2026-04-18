from __future__ import annotations

import re

import pytest

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data

data = [1, 3, 2]


@pytest.mark.parametrize(("index", "expected"), [(0, 1), (1, 3)])
def test_item(nw_eager_constructor: ConstructorEager, index: int, expected: int) -> None:
    series = nw.from_native(nw_eager_constructor({"a": data}), eager_only=True)["a"]
    result = series.item(index)
    assert_equal_data({"a": [result]}, {"a": [expected]})
    assert_equal_data({"a": [series.head(1).item()]}, {"a": [1]})

    with pytest.raises(
        ValueError,
        match=re.escape("can only call '.item()' if the Series is of length 1,"),
    ):
        series.item(None)


def test_out_of_range(nw_eager_constructor: ConstructorEager) -> None:
    series = nw.from_native(nw_eager_constructor({"a": [1, 2]}), eager_only=True)["a"]
    with pytest.raises(IndexError, match="out of range"):
        series.item(2)
