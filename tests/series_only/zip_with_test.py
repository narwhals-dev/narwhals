from __future__ import annotations

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data


def test_zip_with(constructor_eager: ConstructorEager) -> None:
    series1 = nw.from_native(constructor_eager({"a": [1, 3, 2]}), eager_only=True)["a"]
    series2 = nw.from_native(constructor_eager({"a": [4, 4, 6]}), eager_only=True)["a"]
    mask = nw.from_native(constructor_eager({"a": [True, False, True]}), eager_only=True)[
        "a"
    ]

    result = series1.zip_with(mask, series2)
    expected = [1, 4, 2]
    assert_equal_data({"a": result}, {"a": expected})


def test_zip_with_length_1(constructor_eager: ConstructorEager) -> None:
    series1 = nw.from_native(constructor_eager({"a": [1]}), eager_only=True)["a"]
    series2 = nw.from_native(constructor_eager({"a": [4]}), eager_only=True)["a"]
    mask = nw.from_native(constructor_eager({"a": [False]}), eager_only=True)["a"]

    result = series1.zip_with(mask, series2)
    expected = [4]
    assert_equal_data({"a": result}, {"a": expected})
