from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError, ShapeError

if TYPE_CHECKING:
    from tests.utils import ConstructorEager

data = {"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"]}


def test_sample_n(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(
        constructor_eager({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"]}), eager_only=True
    )

    result_expr = df.sample(n=2).shape
    expected_expr = (2, 2)
    assert result_expr == expected_expr


def test_sample_default_single_row(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    # Neither `n` nor `fraction`: defaults to a single row.
    assert df.sample().shape == (1, 2)


def test_sample_both_n_and_fraction(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    with pytest.raises(ValueError, match="cannot specify both `n` and `fraction`"):
        df.sample(n=2, fraction=0.5)


def test_sample_negative_size(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    with pytest.raises(
        InvalidOperationError, match="sample size must be a positive integer"
    ):
        df.sample(n=-1)


def test_sample_larger_than_population(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    with pytest.raises(
        ShapeError, match="cannot take a larger sample than the total population"
    ):
        df.sample(n=100)


def test_sample_fraction(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(
        constructor_eager({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"]}), eager_only=True
    )

    result_expr = df.sample(fraction=0.5).shape
    expected_expr = (2, 2)
    assert result_expr == expected_expr


def test_sample_with_seed(constructor_eager: ConstructorEager) -> None:
    size, n = 100, 10
    df = nw.from_native(constructor_eager({"a": range(size)}), eager_only=True)

    r1 = nw.to_native(df.sample(n=n, seed=123))
    r2 = nw.to_native(df.sample(n=n, seed=123))
    r3 = nw.to_native(df.sample(n=n, seed=42))

    assert r1.equals(r2)  # type: ignore[attr-defined]
    assert not r1.equals(r3)  # type: ignore[attr-defined]
