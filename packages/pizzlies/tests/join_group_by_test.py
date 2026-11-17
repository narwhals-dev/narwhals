"""Joins and group-bys checked against Polars, on random data hitting each code path."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Any

import pizzlies as pz
import pytest

if TYPE_CHECKING:
    import narwhals as nw

pl = pytest.importorskip("polars")


def _keys(rng: random.Random, n: int, *, unique: bool, nulls: bool) -> list[Any]:
    values: list[Any] = (
        rng.sample(range(n * 2), n)
        if unique
        else [rng.randrange(n // 3 + 1) for _ in range(n)]
    )
    if nulls:
        values[::5] = [None] * len(values[::5])
    return values


def _rows(df: Any) -> list[tuple[Any, ...]]:
    """Order-insensitive comparison: joins don't guarantee row order."""
    data = df.to_dict(as_series=False) if isinstance(df, pl.DataFrame) else df.to_native()
    return sorted(zip(*data.values(), strict=True), key=repr)


def _assert_same(result: nw.DataFrame[Any], expected: Any) -> None:
    assert result.columns == expected.columns
    assert _rows(result) == _rows(expected)


@pytest.mark.parametrize("how", ["inner", "left", "full", "semi", "anti"])
@pytest.mark.parametrize(
    ("left_unique", "right_unique"),
    [(False, True), (True, False), (False, False), (True, True)],
)
@pytest.mark.parametrize("nulls", [False, True])
@pytest.mark.parametrize("multi_key", [False, True])
@pytest.mark.parametrize(("n_left", "n_right"), [(60, 40), (40, 60)])
def test_join(  # noqa: PLR0917
    how: str,
    left_unique: bool,  # noqa: FBT001
    right_unique: bool,  # noqa: FBT001
    nulls: bool,  # noqa: FBT001
    multi_key: bool,  # noqa: FBT001
    n_left: int,
    n_right: int,
) -> None:
    rng = random.Random(f"{how}{left_unique}{right_unique}{nulls}{multi_key}{n_left}")  # noqa: S311
    left = {
        "k": _keys(rng, n_left, unique=left_unique, nulls=nulls),
        "a": list(range(n_left)),
    }
    right = {
        "k": _keys(rng, n_right, unique=right_unique, nulls=nulls),
        "b": [f"r{i}" for i in range(n_right)],
    }
    on = ["k"]
    if multi_key:
        # A second key that keeps each side's uniqueness: constant, or nullable.
        left["k2"] = [None if nulls and i % 7 == 0 else 1 for i in range(n_left)]
        right["k2"] = [1] * n_right
        on = ["k", "k2"]
    result = pz.from_dict(left).join(pz.from_dict(right), on=on, how=how)  # type: ignore[arg-type]
    expected = pl.DataFrame(left).join(pl.DataFrame(right), on=on, how=how)
    _assert_same(result, expected)


def _group_frame(n_groups: int, *, nulls: bool) -> dict[str, list[Any]]:
    rng = random.Random(n_groups * 2 + nulls)  # noqa: S311
    n = 500
    data: dict[str, list[Any]] = {
        "g": [rng.randrange(n_groups) for _ in range(n)],
        "x": [rng.randrange(-50, 50) for _ in range(n)],
        "y": [rng.random() for _ in range(n)],
        "s": [rng.choice("abcde") for _ in range(n)],
        "o": rng.sample(range(n), n),
    }
    if nulls:
        for name in ("x", "y", "s"):
            data[name] = [None if rng.random() < 0.2 else v for v in data[name]]
    return data


def _first(c: Any, name: str) -> Any:
    """`first` by `o`: Polars spells it `sort_by("o").first()` instead."""
    return c(name).first(order_by="o") if c is pz.col else c(name).sort_by("o").first()


def _last(c: Any, name: str) -> Any:
    return c(name).last(order_by="o") if c is pz.col else c(name).sort_by("o").last()


AGGS = [
    lambda c: c("x").sum(),
    lambda c: c("y").mean(),
    lambda c: c("x").min(),
    lambda c: c("s").max(),
    lambda c: c("x").count(),
    lambda c: c("s").len(),
    lambda c: c("s").n_unique(),
    lambda c: _first(c, "x"),
    lambda c: _last(c, "s"),
    lambda c: c("y").std(),
    lambda c: c("x").median(),
    lambda c: (c("x") * 2 + c("y")).sum().alias("elementwise"),
    lambda c: (c("x") * 2).alias("aliased_before").max(),
    lambda c: c("x").abs().round(0).mean().alias("chained"),
    lambda c: (c("x") - c("x").mean()).abs().sum().alias("not_elementwise"),
    lambda c: c("x").filter(c("y") > 0.5).sum().alias("filtered"),
]


def _aggs(c: Any) -> list[Any]:
    return [agg(c).alias(f"agg_{i}") for i, agg in enumerate(AGGS)]


@pytest.mark.parametrize("n_groups", [3, 200])
@pytest.mark.parametrize("nulls", [False, True])
def test_group_by(n_groups: int, nulls: bool) -> None:  # noqa: FBT001
    data = _group_frame(n_groups, nulls=nulls)
    result = pz.from_dict(data).group_by("g").agg(*_aggs(pz.col)).sort("g")
    expected = pl.DataFrame(data).group_by("g").agg(*_aggs(pl.col)).sort("g")
    assert result.columns == expected.columns
    actual = result.to_native()
    for name, expected_values in expected.to_dict(as_series=False).items():
        for got, want in zip(actual[name], expected_values, strict=True):
            if isinstance(want, float):
                assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-9), name
            else:
                assert got == want, name


def test_group_by_multi_output() -> None:
    data = _group_frame(5, nulls=True)
    del data["s"]
    result = pz.from_dict(data).group_by("g").agg(pz.all().sum()).sort("g")
    expected = pl.DataFrame(data).group_by("g").agg(pl.all().sum()).sort("g")
    assert result.to_native() == expected.to_dict(as_series=False)
