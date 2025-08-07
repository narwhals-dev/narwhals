from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data

data = {
    "left": [1.0, None, float("nan"), float("inf"), float("-inf")],
    "right": [1.005, None, float("nan"), float("inf"), 3.0],
}


@pytest.mark.parametrize(
    ("abs_tol", "rel_tol", "nans_equal", "expected"),
    [
        (0.1, 0.0, False, [True, None, False, True, False]),
        (0.0001, 0.0, True, [False, None, True, True, False]),
        (0.0, 0.1, False, [True, None, False, True, False]),
        (0.0, 0.001, True, [False, None, True, True, False]),
    ],
)
def test_is_close_series_with_series(
    constructor_eager: ConstructorEager,
    abs_tol: float,
    rel_tol: float,
    *,
    nans_equal: bool,
    expected: list[float],
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    left, right = df["left"], df["right"]
    result = left.is_close(right, abs_tol=abs_tol, rel_tol=rel_tol, nans_equal=nans_equal)
    assert_equal_data({"x": result}, {"x": expected})
