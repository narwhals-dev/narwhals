from __future__ import annotations

import narwhals.stable.v1 as nw
from tests.utils import ConstructorEager
from tests.utils import compare_dicts


def test_slice(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [1, 4, 2]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = {"a": df["a"][[0, 1]]}
    expected = {"a": [1, 2]}
    compare_dicts(result, expected)
    result = {"a": df["a"][1:]}
    expected = {"a": [2, 3]}
    compare_dicts(result, expected)
    result = {"b": df[:, 1]}
    expected = {"b": [4, 5, 6]}
    compare_dicts(result, expected)
    result = {"b": df[:, "b"]}
    expected = {"b": [4, 5, 6]}
    compare_dicts(result, expected)
    result = {"b": df[:2, "b"]}
    expected = {"b": [4, 5]}
    compare_dicts(result, expected)
    result = {"b": df[:2, 1]}
    expected = {"b": [4, 5]}
    compare_dicts(result, expected)
    result = {"b": df[[0, 1], 1]}
    expected = {"b": [4, 5]}
    compare_dicts(result, expected)
    result = {"b": df[[], 1]}
    expected = {"b": []}
    compare_dicts(result, expected)
