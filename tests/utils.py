from __future__ import annotations

import math
from typing import Any
from typing import Iterator
from typing import Sequence


def zip_longest(left: Sequence[Any], right: Sequence[Any]) -> Iterator[Any]:
    if len(left) != len(right):
        raise ValueError(
            "left len != right len", len(left), len(right)
        )  # pragma: no cover
    return zip(left, right)


def compare_dicts(result: Any, expected: dict[str, Any]) -> None:
    if hasattr(result, "collect"):
        result = result.collect()
    if hasattr(result, "columns"):
        for key in result.columns:
            assert key in expected
    for key in expected:
        for lhs, rhs in zip_longest(result[key], expected[key]):
            if isinstance(lhs, float) and not math.isnan(lhs):
                assert math.isclose(lhs, rhs, rel_tol=0, abs_tol=1e-6), (lhs, rhs)
            elif isinstance(lhs, float) and math.isnan(lhs):
                assert math.isnan(rhs), (lhs, rhs)  # pragma: no cover
            else:
                assert lhs == rhs, (lhs, rhs)
