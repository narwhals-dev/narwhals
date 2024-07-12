from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw

data = [1, 2, 3]


@pytest.mark.parametrize(
    ("attr", "rhs", "expected"),
    [
        ("__add__", 1, [2, 3, 4]),
        ("__sub__", 1, [0, 1, 2]),
        ("__mul__", 2, [2, 4, 6]),
        ("__truediv__", 2, [0.5, 1.0, 1.5]),
        ("__floordiv__", 2, [0, 1, 1]),
        ("__mod__", 2, [1, 0, 1]),
        ("__pow__", 2, [1, 4, 9]),
    ],
)
def test_arithmetic(
    attr: str, rhs: Any, expected: list[Any], constructor_series: Any, request: Any
) -> None:
    if "pyarrow" in str(constructor_series) and attr == "__mod__":
        request.applymarker(pytest.mark.xfail)
    s = nw.from_native(constructor_series(data), series_only=True)
    result = getattr(s, attr)(rhs)
    assert result.to_numpy().tolist() == expected


@pytest.mark.parametrize(
    ("attr", "rhs", "expected"),
    [
        ("__radd__", 1, [2, 3, 4]),
        ("__rsub__", 1, [0, -1, -2]),
        ("__rmul__", 2, [2, 4, 6]),
        ("__rfloordiv__", 2, [2, 1, 0]),
        ("__rmod__", 2, [0, 0, 2]),
        ("__rpow__", 2, [2, 4, 8]),
    ],
)
def test_rarithmetic(
    attr: str, rhs: Any, expected: list[Any], constructor_series: Any, request: Any
) -> None:
    if "pyarrow" in str(constructor_series) and attr == "__rmod__":
        request.applymarker(pytest.mark.xfail)
    s = nw.from_native(constructor_series(data), series_only=True)
    result = getattr(s, attr)(rhs)
    assert result.to_numpy().tolist() == expected
