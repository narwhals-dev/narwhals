from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError, ShapeError

if TYPE_CHECKING:
    from tests.utils import ConstructorEager

    class SampleKwargs(TypedDict, total=False):
        n: int | None
        fraction: float | None


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
    df = nw.from_native(constructor_eager(data), eager_only=True)

    with pytest.raises(expected_exception=expected_exception, match=msg):
        df.sample(**params)


def test_sample_fraction(constructor_eager: ConstructorEager) -> None:
    df = constructor_eager({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"]})

    result_expr = df.sample(fraction=0.5).shape
    expected_expr = (2, 2)
    assert result_expr == expected_expr


def test_sample_with_seed(constructor_eager: ConstructorEager) -> None:
    size, n = 100, 10
    df = constructor_eager({"a": range(size)})

    r1 = nw.to_native(df.sample(n=n, seed=123))
    r2 = nw.to_native(df.sample(n=n, seed=123))
    r3 = nw.to_native(df.sample(n=n, seed=42))

    assert r1.equals(r2)
    assert not r1.equals(r3)
