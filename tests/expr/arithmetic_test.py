from __future__ import annotations

from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


@pytest.mark.parametrize(
    ("attr", "rhs", "expected"),
    [
        ("__add__", 1, [2, 3, 4]),
        ("__sub__", 1, [0, 1, 2]),
        ("__mul__", 2, [2, 4, 6]),
        ("__truediv__", 2.0, [0.5, 1.0, 1.5]),
        ("__floordiv__", 2, [0, 1, 1]),
        ("__mod__", 2, [1, 0, 1]),
        ("__pow__", 2, [1, 4, 9]),
    ],
)
def test_arithmetic(
    attr: str, rhs: Any, expected: list[Any], constructor_with_pyarrow: Any, request: Any
) -> None:
    if "pandas_pyarrow" in str(constructor_with_pyarrow) and attr == "__mod__":
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 2, 3]}
    df = nw.from_native(constructor_with_pyarrow(data))
    result = df.select(getattr(nw.col("a"), attr)(rhs))
    compare_dicts(result, {"a": expected})
