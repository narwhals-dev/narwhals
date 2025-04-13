from __future__ import annotations

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


def test_expr_sample(constructor_eager: ConstructorEager) -> None:
    df = nw_v1.from_native(
        constructor_eager({"a": [1, 2, 3], "b": [4, 5, 6]}), eager_only=True
    )

    result_expr = df.select(nw_v1.col("a").sample(n=2)).shape
    expected_expr = (2, 1)
    assert result_expr == expected_expr

    result_series = df["a"].sample(n=2).shape
    expected_series = (2,)
    assert result_series == expected_series

    with pytest.deprecated_call(
        match="is deprecated and will be removed in a future version"
    ):
        df.select(nw.col("a").sample(n=2))


def test_expr_sample_fraction(constructor_eager: ConstructorEager) -> None:
    df = nw_v1.from_native(
        constructor_eager({"a": [1, 2, 3] * 10, "b": [4, 5, 6] * 10}), eager_only=True
    )

    result_expr = df.select(nw_v1.col("a").sample(fraction=0.1)).shape
    expected_expr = (3, 1)
    assert result_expr == expected_expr

    result_series = df["a"].sample(fraction=0.1).shape
    expected_series = (3,)
    assert result_series == expected_series


def test_sample_with_seed(constructor_eager: ConstructorEager) -> None:
    size, n = 100, 10
    df = nw_v1.from_native(constructor_eager({"a": list(range(size))}))
    expected = {"res1": [True], "res2": [False]}
    result = df.select(
        seed1=nw_v1.col("a").sample(n=n, seed=123),
        seed2=nw_v1.col("a").sample(n=n, seed=123),
        seed3=nw_v1.col("a").sample(n=n, seed=42),
    ).select(
        res1=(nw_v1.col("seed1") == nw_v1.col("seed2")).all(),
        res2=(nw_v1.col("seed1") == nw_v1.col("seed3")).all(),
    )

    assert_equal_data(result, expected)

    series = df["a"]
    seed1 = series.sample(n=n, seed=123)
    seed2 = series.sample(n=n, seed=123)
    seed3 = series.sample(n=n, seed=42)

    assert_equal_data(
        {"res1": [(seed1 == seed2).all()], "res2": [(seed1 == seed3).all()]}, expected
    )
