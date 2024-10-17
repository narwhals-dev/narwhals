from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = [1, 2, 3]


@pytest.mark.parametrize("sep", ["_", "-"])
def test_to_dummies(constructor_eager: Any, sep: str) -> None:
    s = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"].alias("a")
    result = s.to_dummies(separator=sep)
    expected = {f"a{sep}1": [1, 0, 0], f"a{sep}2": [0, 1, 0], f"a{sep}3": [0, 0, 1]}

    compare_dicts(result, expected)


@pytest.mark.parametrize("sep", ["_", "-"])
def test_to_dummies_drop_first(constructor_eager: Any, sep: str) -> None:
    s = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"].alias("a")
    result = s.to_dummies(drop_first=True, separator=sep)
    expected = {f"a{sep}2": [0, 1, 0], f"a{sep}3": [0, 0, 1]}

    compare_dicts(result, expected)
