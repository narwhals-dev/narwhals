"""Edge cases where `pizzlies` (eager and lazy) must match Polars, through Narwhals.

Each case was a bug, found by fuzzing against Polars or by porting Polars' own tests.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pizzlies as pz
import pytest

import narwhals as nw

if TYPE_CHECKING:
    from collections.abc import Callable

pl = pytest.importorskip("polars")


def _outcome(frame: Any, query: Callable[[Any], Any], *, unordered: bool) -> Any:
    try:
        result = query(frame)
        if isinstance(result, nw.LazyFrame):
            result = result.collect()
        data = result.to_dict(as_series=False)
    except Exception as exc:  # noqa: BLE001
        return type(exc).__name__
    # NaN != NaN, so compare reprs.
    if unordered:
        rows = sorted(map(repr, zip(*data.values(), strict=True)))
        return list(data), rows
    return repr(data)


def _check(
    data: dict[str, list[Any]], query: Callable[[Any], Any], *, unordered: bool = False
) -> None:
    expected = _outcome(nw.from_native(pl.DataFrame(data)), query, unordered=unordered)
    for frame in (pz.from_dict(data), pz.from_dict(data).lazy()):
        outcome = _outcome(frame, query, unordered=unordered)
        if isinstance(expected, str) and expected.endswith("Error"):
            # Polars' own exception types (e.g. `SchemaError`) aren't Narwhals'.
            assert isinstance(outcome, str), outcome
            assert outcome.endswith("Error"), outcome
        else:
            assert outcome == expected


@pytest.mark.parametrize(
    ("data", "query"),
    [
        ({"a": [1, -3, None]}, lambda df: df.select(nw.col("a") % 0)),
        ({"a": [1.5, -3.0]}, lambda df: df.select(nw.col("a") % 0.0)),
        ({"a": [0, 2]}, lambda df: df.select(nw.lit(5) % nw.col("a"))),
        ({"a": [5.0, -5.0]}, lambda df: df.select(nw.col("a") % math.inf)),
        ({"a": [-0.0, 0.5, -0.5]}, lambda df: df.select(nw.col("a") // 1.0)),
        ({"a": [1, None, 3]}, lambda df: df.select(nw.col("a").clip(None, None))),
        (
            {"a": [2, 5], "lo": [1, None], "hi": [None, 4]},
            lambda df: df.select(nw.col("a").clip(nw.col("lo"), nw.col("hi"))),
        ),
        ({"a": [-0.5, 2.5, 0.5]}, lambda df: df.select(nw.col("a").clip(0, 1))),
        (
            {"a": [3.0, math.inf, 1.0]},
            lambda df: df.select(nw.col("a").quantile(1.0, "linear")),
        ),
        ({"a": [-math.inf]}, lambda df: df.select(nw.col("a").median())),
        ({"a": [1, 300]}, lambda df: df.select(nw.col("a").cast(nw.Int8))),
        ({"a": [-1]}, lambda df: df.select(nw.col("a").cast(nw.UInt8))),
        ({"a": [math.nan, 1.5]}, lambda df: df.select(nw.col("a").cast(nw.String))),
        ({"a": ["true"]}, lambda df: df.select(nw.col("a").cast(nw.Boolean))),
        ({"a": [[1]]}, lambda df: df.select(nw.col("a").cast(nw.String))),
        ({"a": [math.inf]}, lambda df: df.select(nw.col("a").cast(nw.Int64))),
        ({"a": [1]}, lambda df: df.select(nw.col("a").sum().over("zz"))),
        ({"a": [1]}, lambda df: df.select(nw.nth(3))),
        ({"a": [1]}, lambda df: nw.concat([df, df.rename({"a": "b"})])),
        (
            {"a": [1]},
            lambda df: nw.concat([df, df.with_columns(nw.col("a").cast(nw.String))]),
        ),
        (
            {"a": [1]},
            lambda df: nw.concat(
                [df, df.with_columns(nw.col("a").cast(nw.Float64))], how="diagonal"
            ),
        ),
        ({"a": [True], "b": [5]}, lambda df: df.unpivot(["a", "b"])),
        ({"a": [1], "b": [2.5]}, lambda df: df.unpivot(["a", "b"])),
        ({"a": [1], "b": ["x"]}, lambda df: df.unpivot(["a", "b"])),
        (
            {"a": [1]},
            lambda df: df.with_columns(nw.lit(1).alias("z"), nw.lit(2).alias("z")),
        ),
        (
            {"t": [1, 2, 3]},
            lambda df: df.join_asof(
                df.rename({"t": "u"}).with_columns(v=nw.col("u") * 10),
                left_on="t",
                right_on="u",
            ),
        ),
        (
            {"t": [1, 2], "u": [0, 0]},
            lambda df: df.join_asof(
                df.select(nw.col("t").alias("u"), nw.col("t").alias("v")),
                left_on="t",
                right_on="u",
            ),
        ),
        (
            {"a": [[1, None], [1, 2], [None], [], [0, 5], None, [1]]},
            lambda df: df.sort("a"),
        ),
        (
            {"a": [[1, None], [1, 2], [None], [], [0, 5], None, [1]]},
            lambda df: df.sort("a", descending=True),
        ),
        (
            {"s": [{"x": 1, "y": None}, {"x": 1, "y": 2}, {"x": None, "y": 0}]},
            lambda df: df.sort("s"),
        ),
        ({"c": []}, lambda df: df.with_columns(nw.col("c").max().over("c"))),
        ({"a": [[1], [2]], "b": [[], []]}, lambda df: df.unpivot(["a", "b"])),
        (
            {"a": [1, 2], "b": [True, None]},
            lambda df: df.select(
                nw.col("a").fill_null(-1.5),
                nw.col("b").fill_null(1.5),
                w=nw.when(nw.col("a") > 0).then(nw.col("a")).otherwise(2.5),
            ),
        ),
    ],
)
def test_matches_polars(data: dict[str, list[Any]], query: Callable[[Any], Any]) -> None:
    _check(data, query)


NESTED = {"x": [[1], [1], [2], None, [1, None]], "y": [1, 2, 3, 4, 5]}
STRUCTS = {
    "s": [{"x": 1, "y": None}, {"x": 1, "y": 2}, {"x": None, "y": 0}, {"x": 1, "y": 2}],
    "v": [1, 2, 3, 4],
}


@pytest.mark.parametrize(
    ("data", "query"),
    [
        (NESTED, lambda df: df.unique("x", keep="any").select("x")),
        (NESTED, lambda df: df.group_by("x").agg(nw.col("y").sum())),
        (NESTED, lambda df: df.join(df, on="x")),
        (NESTED, lambda df: df.select(nw.col("x").n_unique())),
        (NESTED, lambda df: df.select(nw.col("y").sum().over("x"))),
        (STRUCTS, lambda df: df.unique("s", keep="any").select("s")),
        (STRUCTS, lambda df: df.group_by("s").agg(nw.col("v").sum())),
        (
            {"k": [[]], "v": [1]},
            lambda df: df.join(df.with_columns(w=nw.col("v") + 1), on="k"),
        ),
    ],
)
def test_nested_keys_match_polars(
    data: dict[str, list[Any]], query: Callable[[Any], Any]
) -> None:
    _check(data, query, unordered=True)


@pytest.mark.parametrize("strategy", ["backward", "forward", "nearest"])
def test_join_asof_nan_keys_match_polars(strategy: str) -> None:
    left = {"t": [1.0, math.nan, 5.0, 50.0], "id": [1, 2, 3, 4]}
    right = {"t": [0.0, 5.0, 10.0, math.nan], "v": [100, 200, 300, 400]}

    def query(df: Any) -> Any:
        if df.implementation.is_polars():
            other = nw.from_native(pl.DataFrame(right))
        else:
            other = pz.from_dict(right)
            other = other.lazy() if isinstance(df, nw.LazyFrame) else other
        return df.sort("t").join_asof(other.sort("t"), on="t", strategy=strategy)

    _check(left, lambda df: query(df).sort("id"))


def test_group_iteration_yields_original_keys() -> None:
    df = pz.from_dict(NESTED)
    keys = sorted((key for key, _ in df.group_by("x")), key=repr)
    assert keys == sorted([([1],), ([2],), (None,), ([1, None],)], key=repr)
