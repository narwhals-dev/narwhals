from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, assert_equal_data

data = {"alpha": [1.0, 2.0, None, 4.0], "beta gamma": [None, 3.0, None, 5.0]}


def test_drop_nulls(constructor: Constructor) -> None:
    result = nw.from_native(constructor(data)).drop_nulls()
    expected = {"alpha": [2.0, 4.0], "beta gamma": [3.0, 5.0]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("subset", "expected"),
    [
        ("alpha", {"alpha": [1, 2.0, 4.0], "beta gamma": [None, 3.0, 5.0]}),
        (["alpha"], {"alpha": [1, 2.0, 4.0], "beta gamma": [None, 3.0, 5.0]}),
        (["alpha", "beta gamma"], {"alpha": [2.0, 4.0], "beta gamma": [3.0, 5.0]}),
    ],
)
def test_drop_nulls_subset(
    constructor: Constructor, subset: str | list[str], expected: dict[str, float]
) -> None:
    result = nw.from_native(constructor(data)).drop_nulls(subset=subset)
    assert_equal_data(result, expected)
