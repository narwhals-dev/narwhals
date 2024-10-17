from typing import Any

import polars as pl
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import compare_dicts


@pytest.mark.parametrize("expr1", ["a", nw.col("a")])
@pytest.mark.parametrize("expr2", ["b", nw.col("b")])
def test_allh(constructor: Constructor, expr1: Any, expr2: Any) -> None:
    data = {
        "a": [False, False, True],
        "b": [False, True, True],
    }
    df = nw.from_native(constructor(data))
    result = df.select(all=nw.all_horizontal(expr1, expr2))

    expected = {"all": [False, False, True]}
    compare_dicts(result, expected)


def test_allh_series(constructor_eager: ConstructorEager) -> None:
    data = {
        "a": [False, False, True],
        "b": [False, True, True],
    }
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(all=nw.all_horizontal(df["a"], df["b"]))

    expected = {"all": [False, False, True]}
    compare_dicts(result, expected)


def test_allh_all(constructor: Constructor) -> None:
    data = {
        "a": [False, False, True],
        "b": [False, True, True],
    }
    df = nw.from_native(constructor(data))
    result = df.select(all=nw.all_horizontal(nw.all()))
    expected = {"all": [False, False, True]}
    compare_dicts(result, expected)
    result = df.select(nw.all_horizontal(nw.all()))
    expected = {"a": [False, False, True]}
    compare_dicts(result, expected)


def test_allh_nth(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "polars" in str(constructor) and parse_version(pl.__version__) < (1, 0):
        request.applymarker(pytest.mark.xfail)
    data = {
        "a": [False, False, True],
        "b": [False, True, True],
    }
    df = nw.from_native(constructor(data))
    result = df.select(nw.all_horizontal(nw.nth(0, 1)))
    expected = {"a": [False, False, True]}
    compare_dicts(result, expected)
    result = df.select(nw.all_horizontal(nw.col("a"), nw.nth(0)))
    expected = {"a": [False, False, True]}
    compare_dicts(result, expected)


def test_horizontal_expressions_emtpy(constructor: Constructor) -> None:
    data = {
        "a": [False, False, True],
        "b": [False, True, True],
    }
    df = nw.from_native(constructor(data))
    with pytest.raises(
        ValueError, match=r"At least one expression must be passed.*all_horizontal"
    ):
        df.select(nw.all_horizontal())
    with pytest.raises(
        ValueError, match=r"At least one expression must be passed.*any_horizontal"
    ):
        df.select(nw.any_horizontal())
    with pytest.raises(
        ValueError, match=r"At least one expression must be passed.*mean_horizontal"
    ):
        df.select(nw.mean_horizontal())
    with pytest.raises(
        ValueError, match=r"At least one expression must be passed.*sum_horizontal"
    ):
        df.select(nw.sum_horizontal())

    with pytest.raises(
        ValueError, match=r"At least one expression must be passed.*max_horizontal"
    ):
        df.select(nw.max_horizontal())

    with pytest.raises(
        ValueError, match=r"At least one expression must be passed.*min_horizontal"
    ):
        df.select(nw.min_horizontal())
