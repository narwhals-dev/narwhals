from __future__ import annotations

from typing import Any
from typing import Iterator
from typing import Sequence

import polars as pl


def zip_longest(left: Sequence[Any], right: Sequence[Any]) -> Iterator[Any]:
    if len(left) != len(right):
        raise ValueError("left len != right len", len(left), len(right))
    return zip(left, right)


def compare_dicts(result: dict[str, Any], expected: dict[str, Any]) -> None:
    if isinstance(result, pl.LazyFrame):
        result = result.collect()
    for key in expected:
        for lhs, rhs in zip_longest(result[key], expected[key]):
            if isinstance(lhs, float):
                assert abs(lhs - rhs) < 1e-6, (lhs, rhs)
            else:
                assert lhs == rhs, (lhs, rhs)
