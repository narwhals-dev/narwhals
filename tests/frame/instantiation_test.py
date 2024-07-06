from typing import Any

import pytest

import narwhals as nw
from tests.utils import compare_dicts


def test_lazy_instantiation(constructor: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    result = nw.from_native(constructor(data))
    expected = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    compare_dicts(result, expected)


def test_lazy_instantiation_error(constructor: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    with pytest.raises(
        TypeError, match="Can't instantiate DataFrame from Polars LazyFrame."
    ):
        _ = nw.DataFrame(constructor(data)).shape


def test_eager_instantiation(constructor: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    result = nw.from_native(constructor(data), eager_only=True)
    expected = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    compare_dicts(result, expected)
