"""NaN semantics, checked against Polars.

Separate `float("nan")` objects are deliberately used throughout: Python only finds a NaN
in a dict or set by identity, so a single shared NaN object would hide the bugs.
"""

from __future__ import annotations

from typing import Any

import pizzlies as pz
import pytest

pl = pytest.importorskip("polars")


def nans(n: int) -> list[float]:
    return [float("nan") for _ in range(n)]


def as_dict(df: Any) -> Any:
    return df.to_dict(as_series=False) if isinstance(df, pl.DataFrame) else df.to_native()


def assert_same(result: Any, expected: Any) -> None:
    # `repr`, because NaN != NaN.
    assert repr(result) == repr(expected)


@pytest.mark.parametrize(
    "method",
    [
        lambda s: s.n_unique(),
        lambda s: sorted(s.unique().to_list(), key=repr),
        lambda s: s.is_unique().to_list(),
        lambda s: s.is_first_distinct().to_list(),
        lambda s: s.is_last_distinct().to_list(),
        lambda s: s.is_in([float("nan")]).to_list(),
        lambda s: s.value_counts(sort=True)["count"].to_list(),
        lambda s: s.mode().to_list(),
    ],
)
def test_nans_are_equal(method: Any) -> None:
    values = [*nans(2), 1.0, None]
    assert_same(method(pz.new_series("a", values)), method(pl.Series("a", values)))


def test_frame_nans_are_equal() -> None:
    data = {"g": [*nans(2), 1.0], "x": [1, 2, 3]}
    df, expected = pz.from_dict(data), pl.DataFrame(data)
    assert_same(
        as_dict(df.group_by("g").agg(pz.col("x").sum()).sort("g")),
        as_dict(expected.group_by("g").agg(pl.col("x").sum()).sort("g")),
    )
    assert_same(
        df.with_columns(s=pz.col("x").sum().over("g"))["s"].to_list(),
        expected.with_columns(s=pl.col("x").sum().over("g"))["s"].to_list(),
    )
    assert_same(
        df.with_columns(s=pz.col("x").cum_sum().over("g", order_by="x"))["s"].to_list(),
        expected.with_columns(s=pl.col("x").cum_sum().over("g", order_by="x"))[
            "s"
        ].to_list(),
    )
    assert df.unique(["g"]).shape == expected.unique(["g"]).shape
    assert (
        df.select("g").is_unique().to_list() == expected.select("g").is_unique().to_list()
    )
    right = {"g": nans(1), "y": [4]}
    assert_same(
        as_dict(df.join(pz.from_dict(right), on="g")),
        as_dict(expected.join(pl.DataFrame(right), on="g")),
    )


@pytest.mark.parametrize(
    "values",
    [
        [float("nan"), 1.0, 2.0],
        [1.0, float("nan"), 2.0],
        [1.0, 2.0, float("nan")],
        [None, float("nan"), 1.0],
        nans(2),
    ],
)
def test_min_max_skip_nan(values: list[Any]) -> None:
    def results(s: Any) -> tuple[Any, ...]:
        return (s.min(), s.max(), s.arg_min(), s.arg_max())

    assert_same(results(pz.new_series("a", values)), results(pl.Series("a", values)))


@pytest.mark.parametrize("with_nulls", [False, True])
def test_min_max_skip_nan_in_group_by_and_horizontal(with_nulls: bool) -> None:  # noqa: FBT001
    data: dict[str, list[Any]] = {
        "g": [1, 1, 2, 2],
        "a": [*nans(1), 1.0, *nans(2)],
        "b": [2.0, *nans(2), 3.0],
    }
    if with_nulls:
        data["a"][1] = None
    df, expected = pz.from_dict(data), pl.DataFrame(data)
    assert_same(
        as_dict(df.group_by("g").agg(pz.col("a").min(), pz.col("b").max()).sort("g")),
        as_dict(
            expected.group_by("g").agg(pl.col("a").min(), pl.col("b").max()).sort("g")
        ),
    )
    assert_same(
        as_dict(df.select(pz.min_horizontal("a", "b"), m=pz.max_horizontal("a", "b"))),
        as_dict(
            expected.select(pl.min_horizontal("a", "b"), m=pl.max_horizontal("a", "b"))
        ),
    )
