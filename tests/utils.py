from __future__ import annotations

from typing import Any

import polars as pl


def compare_dicts(result: dict[str, Any], expected: dict[str, Any]) -> None:
    if isinstance(result, pl.LazyFrame):
        result = result.collect()
    for key in expected:
        for lhs, rhs in zip(result[key], expected[key]):
            if isinstance(lhs, float):
                assert abs(lhs - rhs) < 1e-6
            else:
                assert lhs == rhs
