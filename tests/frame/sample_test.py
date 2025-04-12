from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


def test_sample_n(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(
        constructor_eager({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"]}), eager_only=True
    )

    result_expr = df.sample(n=2).shape
    expected_expr = (2, 2)
    assert result_expr == expected_expr


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

    assert r1.equals(r2)  # type: ignore[union-attr]
    assert not r1.equals(r3)  # type: ignore[union-attr]
