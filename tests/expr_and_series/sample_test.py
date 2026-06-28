from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError, ShapeError
from tests.utils import ConstructorEager, assert_equal_data

if TYPE_CHECKING:

    class SampleKwargs(TypedDict, total=False):
        n: int | None
        fraction: float | None


data = {"a": [1, 2, 3, 4]}


def test_sample_fraction(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(
        constructor_eager({"a": [1, 2, 3] * 10, "b": [4, 5, 6] * 10}), eager_only=True
    )

    result_series = df["a"].sample(fraction=0.1).shape
    expected_series = (3,)
    assert result_series == expected_series


def test_sample_default_single_row(constructor_eager: ConstructorEager) -> None:
    s = nw.from_native(constructor_eager(data), eager_only=True)["a"]

    # Neither `n` nor `fraction`: defaults to a single item.
    assert s.sample().shape == (1,)


@pytest.mark.parametrize(
    ("expected_exception", "msg", "params"),
    [
        (ValueError, "cannot specify both `n` and `fraction`", {"n": 2, "fraction": 0.5}),
        (InvalidOperationError, "sample size must be a positive integer", {"n": -1}),
        (ShapeError, "cannot take a larger sample than the total population", {"n": 10}),
    ],
)
def test_sample_raises(
    constructor_eager: ConstructorEager,
    expected_exception: type[Exception],
    msg: str,
    params: SampleKwargs,
) -> None:
    s = nw.from_native(constructor_eager(data), eager_only=True)["a"]

    with pytest.raises(expected_exception=expected_exception, match=msg):
        s.sample(**params)


def test_sample_with_seed(constructor_eager: ConstructorEager) -> None:
    size, n = 100, 10
    df = nw.from_native(constructor_eager({"a": list(range(size))}))
    expected = {"res1": [True], "res2": [False]}
    series = df["a"]
    seed1 = series.sample(n=n, seed=123)
    seed2 = series.sample(n=n, seed=123)
    seed3 = series.sample(n=n, seed=42)
    assert_equal_data(
        {"res1": [(seed1 == seed2).all()], "res2": [(seed1 == seed3).all()]}, expected
    )
