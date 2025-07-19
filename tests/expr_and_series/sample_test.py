from __future__ import annotations

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data


def test_sample_fraction(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(
        constructor_eager({"a": [1, 2, 3] * 10, "b": [4, 5, 6] * 10}), eager_only=True
    )

    result_series = df["a"].sample(fraction=0.1).shape
    expected_series = (3,)
    assert result_series == expected_series


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
