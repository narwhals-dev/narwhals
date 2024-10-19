from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import compare_dicts

data = {
    "a": [1.0, 2.0, None, 4.0],
    "b": [None, 3.0, None, 5.0],
}


def test_drop_nulls(constructor: Constructor) -> None:
    result = nw.from_native(constructor(data)).drop_nulls()
    expected = {
        "a": [2.0, 4.0],
        "b": [3.0, 5.0],
    }
    compare_dicts(result, expected)


@pytest.mark.parametrize(
    ("subset", "expected"),
    [
        ("a", {"a": [1, 2.0, 4.0], "b": [float("nan"), 3.0, 5.0]}),
        (["a"], {"a": [1, 2.0, 4.0], "b": [float("nan"), 3.0, 5.0]}),
        (["a", "b"], {"a": [2.0, 4.0], "b": [3.0, 5.0]}),
    ],
)
def test_drop_nulls_subset(
    constructor: Constructor, subset: str | list[str], expected: dict[str, float]
) -> None:
    result = nw.from_native(constructor(data)).drop_nulls(subset=subset)
    compare_dicts(result, expected)
