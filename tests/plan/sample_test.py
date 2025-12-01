from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from narwhals.exceptions import ShapeError
from tests.plan.utils import dataframe, series

if TYPE_CHECKING:
    from tests.conftest import Data


@pytest.fixture(scope="module")
def data() -> Data:
    return {"a": [1, 2, 3] * 10, "b": [4, 5, 6] * 10}


@pytest.fixture(scope="module")
def data_big() -> Data:
    return {"a": list(range(100))}


@pytest.mark.parametrize("n", [None, 1, 7, 29])
def test_sample_n_series(data: Data, n: int | None) -> None:
    result = series(data["a"]).sample(n).shape
    expected = (1,) if n is None else (n,)
    assert result == expected


def test_sample_fraction_series(data: Data) -> None:
    result = series(data["a"]).sample(fraction=0.1).shape
    expected = (3,)
    assert result == expected


@pytest.mark.parametrize("n", [10])
def test_sample_with_seed_series(data_big: Data, n: int) -> None:
    ser = series(data_big["a"])
    seed1 = ser.sample(n, seed=123)
    seed2 = ser.sample(n, seed=123)
    seed3 = ser.sample(n, seed=42)
    result = {"res1": [(seed1 == seed2).all()], "res2": [(seed1 == seed3).all()]}
    expected = {"res1": [True], "res2": [False]}
    assert result == expected


@pytest.mark.parametrize("n", [2, None, 1, 18])
def test_sample_n_dataframe(data: Data, n: int | None) -> None:
    result = dataframe(data).sample(n=n).shape
    expected = (1, 2) if n is None else (n, 2)
    assert result == expected


def test_sample_fraction_dataframe(data: Data) -> None:
    result = dataframe(data).sample(fraction=0.5).shape
    expected = (15, 2)
    assert result == expected


@pytest.mark.parametrize("n", [10])
def test_sample_with_seed_dataframe(data_big: Data, n: int) -> None:
    df = dataframe(data_big)
    r1 = df.sample(n, seed=123).to_native()
    r2 = df.sample(n, seed=123).to_native()
    r3 = df.sample(n, seed=42).to_native()
    assert r1.equals(r2)
    assert not r1.equals(r3)


@pytest.mark.parametrize("n", [39, 42, 20, 99])
def test_sample_with_replacement_series(data: Data, n: int) -> None:
    result = series(data["a"]).slice(0, 10).sample(n, with_replacement=True)
    assert len(result) == n


@pytest.mark.parametrize("n", [10, 15, 28, 100])
def test_sample_with_replacement_dataframe(data: Data, n: int) -> None:
    result = dataframe(data).slice(0, 5).sample(n, with_replacement=True)
    assert len(result) == n


def test_sample_invalid(data: Data) -> None:
    df = dataframe(data)
    ser = df.to_series()

    both_n_fraction = r"cannot specify both `n` and `fraction`"
    too_high_n = r"cannot take a larger sample than the total population when `with_replacement=false`"

    with pytest.raises(ValueError, match=both_n_fraction):
        df.sample(n=1, fraction=0.5)
    with pytest.raises(ValueError, match=both_n_fraction):
        ser.sample(n=567, fraction=0.1)
    with pytest.raises(ShapeError, match=too_high_n):
        df.sample(n=1_000)
    with pytest.raises(ShapeError, match=too_high_n):
        ser.sample(n=2_000)
