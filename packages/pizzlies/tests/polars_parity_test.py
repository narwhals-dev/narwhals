"""Edge cases where `pizzlies` (eager and lazy) must match Polars, through Narwhals.

Each case was a bug, found by fuzzing against Polars or by porting Polars' own tests.
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pizzlies as pz
import pytest

import narwhals as nw
import narwhals.selectors as ncs
from narwhals.exceptions import ColumnNotFoundError, InvalidOperationError

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
        ({"a": [1, None, 3]}, lambda df: df.select(nw.col("a").clip(None, None))),
        (
            {"a": [2, 5], "lo": [1, None], "hi": [None, 4]},
            lambda df: df.select(nw.col("a").clip(nw.col("lo"), nw.col("hi"))),
        ),
        (
            {"a": [3.0, math.inf, 1.0]},
            lambda df: df.select(nw.col("a").quantile(1.0, "linear")),
        ),
        ({"a": [-math.inf]}, lambda df: df.select(nw.col("a").median())),
        ({"a": [1, 300]}, lambda df: df.select(nw.col("a").cast(nw.Int8))),
        ({"a": [-1]}, lambda df: df.select(nw.col("a").cast(nw.UInt8))),
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
            {"a": ["9223372036854775808"]},
            lambda df: df.select(nw.col("a").cast(nw.Int64)),
        ),
        ({"a": [2**63]}, lambda df: df.select(nw.col("a").cast(nw.Int64))),
        ({"a": [Decimal("1.5")], "b": [True]}, lambda df: df.unpivot(["a", "b"])),
        (
            {"s": [{"a": 1, "b": 9}, {"b": 0, "a": 2}], "i": [0, 1]},
            lambda df: df.sort("s").select("i"),
        ),
        ({"a": [1]}, lambda df: df.select("a", "a")),
        (
            {"k": [1, 2], "c": [True, False]},
            lambda df: df.group_by("k").agg(ncs.numeric().max()).sort("k"),
        ),
        (
            {"k": [300]},
            lambda df: df.group_by("k").agg(ncs.numeric().cast(nw.Int8).max()),
        ),
        (
            {"k": [300], "v": [1]},
            lambda df: df.group_by("k").agg(ncs.numeric().cast(nw.Int8).max()),
        ),
        (
            {"k": [300, 300], "v": [1, 2]},
            lambda df: df.group_by("k").agg(ncs.numeric().cast(nw.Int8).unique().max()),
        ),
        (
            {"c": [1, None], "d": [None, 2], "e": [1, 1]},
            lambda df: df.group_by("c", drop_null_keys=True).agg(ncs.numeric().max()),
        ),
    ],
)
def test_matches_polars(data: dict[str, list[Any]], query: Callable[[Any], Any]) -> None:
    _check(data, query)


@pytest.mark.parametrize(
    "query",
    [
        lambda df: df.group_by("zz").agg(nw.col("a").sum()),
        # Key aliases and temporary key columns aren't input columns.
        lambda df: df.group_by(nw.col("a").alias("k")).agg(nw.col("k").sum()),
        lambda df: df.group_by(nw.col("a").alias("k")).agg(nw.col("_k_tmp").sum()),
        # Even next to, or in, a selector, whose input columns aren't known up front.
        lambda df: df.group_by(nw.col("a").alias("k")).agg(
            ncs.numeric().sum(), nw.col("_k_tmp").sum()
        ),
        lambda df: df.group_by(nw.col("a").alias("k")).agg(
            (ncs.numeric() + nw.col("_k_tmp")).sum()
        ),
    ],
)
def test_group_by_missing_column(query: Callable[[Any], Any]) -> None:
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        query(nw.from_native(pl.DataFrame({"a": [1, 2]})))
    with pytest.raises(ColumnNotFoundError):
        query(pz.from_dict({"a": [1, 2]}))


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
            {"s": [{"a": 1, "b": 2}, {"b": 2, "a": 1}], "v": [1, 2]},
            lambda df: df.group_by("s").agg(nw.col("v").sum()),
        ),
        (
            {"s": [{"a": 1, "b": 2}, {"b": 2, "a": 1}]},
            lambda df: df.unique("s", keep="any"),
        ),
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


@pytest.mark.parametrize(
    "data",
    [
        {"a": [True], "b": [5]},
        {"a": [1], "b": ["x"]},
        {"a": [Decimal("1.5")], "b": [2]},
        {"a": [Decimal("1.5")], "b": [2.5]},
    ],
)
def test_unpivot_mixed_dtypes_raises(data: dict[str, list[Any]]) -> None:
    # Polars casts to a common dtype: pizzlies is stricter, rather than differ.
    pl.DataFrame(data).unpivot(["a", "b"])
    for frame in (pz.from_dict(data), pz.from_dict(data).lazy()):
        with pytest.raises(InvalidOperationError, match="different dtypes"):
            nw.to_native(frame.unpivot(["a", "b"]).lazy().collect())


@pytest.mark.parametrize(
    ("data", "dtype"),
    [
        ({"a": [1], "b": [2.5]}, nw.Float64()),
        ({"a": [Decimal("1.5")], "b": [Decimal("1.25")]}, nw.Decimal(None, 2)),
    ],
)
def test_unpivot_mixed_dtypes_inference_resolves(
    data: dict[str, list[Any]], dtype: Any
) -> None:
    result = pz.from_dict(data).unpivot(["a", "b"])
    assert result.schema["value"] == dtype


@pytest.mark.parametrize("columns", [[[[]], [[1]], [["x"]]], [[[1]], [["x"]], [[]]]])
def test_concat_nested_dtypes_match_polars(columns: list[list[Any]]) -> None:
    def outcome(frames: list[Any]) -> Any:
        try:
            return nw.concat(frames).to_dict(as_series=False)
        except Exception as exc:  # noqa: BLE001
            return type(exc).__name__

    expected = outcome([nw.from_native(pl.DataFrame({"a": a})) for a in columns])
    result = outcome([pz.from_dict({"a": a}) for a in columns])
    if isinstance(expected, str):
        assert isinstance(result, str), result
    else:
        assert result == expected


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
