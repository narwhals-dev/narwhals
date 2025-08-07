from __future__ import annotations

import pytest

import narwhals as nw
from narwhals.exceptions import ComputeError, InvalidOperationError
from tests.conftest import modin_constructor, pandas_constructor
from tests.utils import ConstructorEager, assert_equal_data

NON_NULLABLE_CONSTRUCTORS = (pandas_constructor, modin_constructor)
NULL_PLACEHOLDER = 999.0
NAN_PLACEHOLDER = -1.0
INF = float("inf")

data = {
    "left": [1.001, NULL_PLACEHOLDER, NAN_PLACEHOLDER, INF, -INF, INF],
    "right": [1.005, NULL_PLACEHOLDER, NAN_PLACEHOLDER, INF, 3.0, -INF],
    "non_numeric": list("number"),
}


def test_is_close_series_raise_non_numeric(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    left, right = df["non_numeric"], df["right"]

    msg = "is_close operation not supported for dtype"
    with pytest.raises(InvalidOperationError, match=msg):
        left.is_close(right)


def test_is_close_series_raise_negative_abs_tol(
    constructor_eager: ConstructorEager,
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    left, right = df["left"], df["right"]

    abs_tol = -2
    msg = rf"`abs_tol` must be non-negative but got {abs_tol}"
    with pytest.raises(ComputeError, match=msg):
        left.is_close(right, abs_tol=abs_tol)

    with pytest.raises(ComputeError, match=msg):
        left.is_close(right, abs_tol=abs_tol, rel_tol=999)


@pytest.mark.parametrize("rel_tol", [-0.0001, 1.0, 1.1])
def test_is_close_series_raise_invalid_rel_tol(
    constructor_eager: ConstructorEager, rel_tol: float
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    left, right = df["left"], df["right"]

    msg = rf"`rel_tol` must be in the range \[0, 1\) but got {rel_tol}"
    with pytest.raises(ComputeError, match=msg):
        left.is_close(right, rel_tol=rel_tol)


@pytest.mark.parametrize(
    ("abs_tol", "rel_tol", "nans_equal", "expected"),
    [
        (0.1, 0.0, False, [True, None, False, True, False, False]),
        (0.0001, 0.0, True, [False, None, True, True, False, False]),
        (0.0, 0.1, False, [True, None, False, True, False, False]),
        (0.0, 0.001, True, [False, None, True, True, False, False]),
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

    nulls = nw.new_series(
        name="nulls",
        values=[None] * len(left),
        dtype=nw.Float64(),
        backend=df.implementation,
    )
    # Tricks to generate nan's and null's for pandas with nullable backends:
    #   * Square rooting a negative number will generate a NaN
    #   * Replacing a value with None once the dtype is nullable will generate <NA>'s
    left = left.zip_with(~left.is_finite(), left**0.5).zip_with(
        left != NULL_PLACEHOLDER, nulls
    )
    right = right.zip_with(~right.is_finite(), right**0.5).zip_with(
        left != NULL_PLACEHOLDER, nulls
    )
    result = left.is_close(right, abs_tol=abs_tol, rel_tol=rel_tol, nans_equal=nans_equal)

    if constructor_eager in NON_NULLABLE_CONSTRUCTORS:
        expected = [v if v is not None else False for v in expected]
    assert_equal_data({"x": result}, {"x": expected})
