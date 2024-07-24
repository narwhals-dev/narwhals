from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = [1, 4, 2, 5]


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        ("left", [True, True, True, False]),
        ("right", [False, True, True, True]),
        ("both", [True, True, True, True]),
        ("none", [False, True, True, False]),
    ],
)
def test_is_between(
    request: Any, constructor_series: Any, closed: str, expected: list[bool]
) -> None:
    if "pandas_series_nullable_constructor" in str(constructor_series):
        request.applymarker(pytest.mark.xfail)

    ser = nw.from_native(constructor_series(data), series_only=True)
    result = ser.is_between(1, 5, closed=closed)
    compare_dicts({"a": result}, {"a": expected})
