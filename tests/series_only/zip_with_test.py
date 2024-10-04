from __future__ import annotations

from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_zip_with(constructor_eager: Any) -> None:
    series1 = nw.from_native(constructor_eager({"a": [1, 3, 2]}), eager_only=True)["a"]
    series2 = nw.from_native(constructor_eager({"a": [4, 4, 6]}), eager_only=True)["a"]
    mask = nw.from_native(constructor_eager({"a": [True, False, True]}), eager_only=True)[
        "a"
    ]

    result = series1.zip_with(mask, series2)
    expected = [1, 4, 2]
    compare_dicts({"a": result}, {"a": expected})


def test_zip_with_length_1(constructor_eager: Any) -> None:
    series1 = nw.from_native(constructor_eager({"a": [1]}), eager_only=True)["a"]
    series2 = nw.from_native(constructor_eager({"a": [4]}), eager_only=True)["a"]
    mask = nw.from_native(constructor_eager({"a": [False]}), eager_only=True)["a"]

    result = series1.zip_with(mask, series2)
    expected = [4]
    compare_dicts({"a": result}, {"a": expected})
