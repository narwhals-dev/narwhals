from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

if TYPE_CHECKING:
    from narwhals.selectors import Selector

data = {
    "a": [1.0, 2.0, None, 4.0],
    "b": [None, "x", None, "y"],
}


def test_drop_nulls(constructor: Any) -> None:
    result = nw.from_native(constructor(data)).drop_nulls()
    expected = {
        "a": [2.0, 4.0],
        "b": ["x", "y"],
    }
    compare_dicts(result, expected)


@pytest.mark.parametrize(
    "subset", ["a", ["a"], nw.selectors.numeric(), [nw.selectors.numeric()]]
)
def test_drop_nulls_subset(
    constructor: Any, subset: str | Selector | list[str | Selector]
) -> None:
    result = nw.from_native(constructor(data)).drop_nulls(subset=subset)
    expected = {
        "a": [1, 2.0, 4.0],
        "b": [float("nan"), "x", "y"],
    }
    compare_dicts(result, expected)
