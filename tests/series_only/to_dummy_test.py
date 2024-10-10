from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = ["x", "y", "z"]
data_na = ["x", "y", None]


@pytest.mark.parametrize("sep", ["_", "-"])
def test_to_dummies(constructor_eager: Any, sep: str) -> None:
    s = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"].alias("a")
    result = s.to_dummies(separator=sep)
    expected = {f"a{sep}x": [1, 0, 0], f"a{sep}y": [0, 1, 0], f"a{sep}z": [0, 0, 1]}

    compare_dicts(result, expected)


@pytest.mark.parametrize("sep", ["_", "-"])
def test_to_dummies_drop_first(
    request: pytest.FixtureRequest, constructor_eager: Any, sep: str
) -> None:
    if "cudf" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
    s = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"].alias("a")
    result = s.to_dummies(drop_first=True, separator=sep)
    expected = {f"a{sep}y": [0, 1, 0], f"a{sep}z": [0, 0, 1]}

    compare_dicts(result, expected)


@pytest.mark.parametrize("sep", ["_", "-"])
def test_to_dummies_with_nulls(constructor_eager: Any, sep: str) -> None:
    if "pandas_nullable_constructor" not in str(constructor_eager):
        pytest.skip()
    s = nw.from_native(constructor_eager({"a": data_na}), eager_only=True)["a"].alias("a")
    result = s.to_dummies(separator=sep)
    expected = {f"a{sep}null": [0, 0, 1], f"a{sep}x": [1, 0, 0], f"a{sep}y": [0, 1, 0]}

    compare_dicts(result, expected)


@pytest.mark.parametrize("sep", ["_", "-"])
def test_to_dummies_drop_first_na(
    request: pytest.FixtureRequest, constructor_eager: Any, sep: str
) -> None:
    if "cudf" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
    s = nw.from_native(constructor_eager({"a": data_na}), eager_only=True)["a"].alias("a")
    result = s.to_dummies(drop_first=True, separator=sep)
    expected = {f"a{sep}null": [0, 0, 1], f"a{sep}y": [0, 1, 0]}

    compare_dicts(result, expected)
