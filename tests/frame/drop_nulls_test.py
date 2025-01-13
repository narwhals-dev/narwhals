from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {
    "a": [1.0, 2.0, None, 4.0],
    "b": [None, 3.0, None, 5.0],
}


def test_drop_nulls(
    constructor: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    result = nw.from_native(constructor(data)).drop_nulls()
    expected = {
        "a": [2.0, 4.0],
        "b": [3.0, 5.0],
    }
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("subset", "expected"),
    [
        ("a", {"a": [1, 2.0, 4.0], "b": [None, 3.0, 5.0]}),
        (["a"], {"a": [1, 2.0, 4.0], "b": [None, 3.0, 5.0]}),
        (["a", "b"], {"a": [2.0, 4.0], "b": [3.0, 5.0]}),
    ],
)
def test_drop_nulls_subset(
    constructor: ConstructorEager,
    subset: str | list[str],
    expected: dict[str, float],
    request: pytest.FixtureRequest,
) -> None:
    if "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    result = nw.from_native(constructor(data)).drop_nulls(subset=subset)
    assert_equal_data(result, expected)
