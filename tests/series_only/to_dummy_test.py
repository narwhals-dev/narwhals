from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data

data = ["x", "y", "z"]
data_na = ["x", "y", None]


@pytest.mark.parametrize("sep", ["_", "-"])
def test_to_dummies(constructor_eager: ConstructorEager, sep: str) -> None:
    s = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"].alias("a")
    result = s.to_dummies(separator=sep)
    expected = {f"a{sep}x": [1, 0, 0], f"a{sep}y": [0, 1, 0], f"a{sep}z": [0, 0, 1]}

    assert_equal_data(result, expected)


@pytest.mark.parametrize("sep", ["_", "-"])
def test_to_dummies_drop_first(constructor_eager: ConstructorEager, sep: str) -> None:
    s = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"].alias("a")
    result = s.to_dummies(drop_first=True, separator=sep)
    expected = {f"a{sep}y": [0, 1, 0], f"a{sep}z": [0, 0, 1]}

    assert_equal_data(result, expected)


@pytest.mark.parametrize("sep", ["_", "-"])
def test_to_dummies_with_nulls(constructor_eager: ConstructorEager, sep: str) -> None:
    if "pandas_nullable_constructor" not in str(constructor_eager):
        pytest.skip()
    s = nw.from_native(constructor_eager({"a": data_na}), eager_only=True)["a"].alias("a")
    result = s.to_dummies(separator=sep)
    expected = {f"a{sep}null": [0, 0, 1], f"a{sep}x": [1, 0, 0], f"a{sep}y": [0, 1, 0]}

    assert_equal_data(result, expected)


@pytest.mark.parametrize("sep", ["_", "-"])
def test_to_dummies_drop_first_na(constructor_eager: ConstructorEager, sep: str) -> None:
    s = nw.from_native(constructor_eager({"a": data_na}), eager_only=True)["a"].alias("a")
    result = s.to_dummies(drop_first=True, separator=sep)
    expected = {f"a{sep}null": [0, 0, 1], f"a{sep}y": [0, 1, 0]}

    assert_equal_data(result, expected)
    assert result.schema == {f"a{sep}null": nw.Int8, f"a{sep}y": nw.Int8}
