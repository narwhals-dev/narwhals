from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING, Any

import pizzlies as pz
import pytest
from pizzlies._plugin import lazy
from pizzlies._plugin.lazy import LazyPlan

import narwhals as nw
from narwhals.exceptions import ColumnNotFoundError

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

DATA = {
    "a": [1, 2, 3, 4],
    "b": ["x", "y", "x", "z"],
    "c": [1.5, 2.5, None, 4.5],
    "d": [True, False, True, None],
}


@pytest.fixture
def path(tmp_path: Path) -> str:
    file = tmp_path / "data.parquet"
    pq.write_table(pa.table(DATA), file)
    return str(file)


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    """The columns read by each Parquet scan."""
    calls: list[tuple[str, ...]] = []
    original = lazy.read_parquet

    def spy(source: Any, columns: Any) -> Any:
        calls.append(tuple(columns))
        return original(source, columns)

    monkeypatch.setattr(lazy, "read_parquet", spy)
    return calls


def check(
    path: str, query: Callable[[Any], Any], *, ordered: bool = True
) -> dict[str, list[Any]]:
    """Run `query` lazily on a scan, and eagerly, and check the results agree."""
    result = query(pz.scan_parquet(path)).collect().to_native()
    expected = query(pz.read_parquet(path)).to_native()
    if ordered:
        assert result == expected
    else:  # e.g. joins don't guarantee row order
        assert list(result) == list(expected)
        assert sorted(zip(*result.values(), strict=True), key=repr) == sorted(
            zip(*expected.values(), strict=True), key=repr
        )
    return result  # type: ignore[no-any-return]


def test_lazy_returns_plan() -> None:
    lf = pz.from_dict(DATA).lazy()
    assert isinstance(lf, pz.LazyFrame)
    assert isinstance(lf.to_native(), LazyPlan)
    assert nw.from_native(lf.to_native()).collect().to_native() == DATA


def test_scan_reads_only_used_columns(path: str, reads: list[tuple[str, ...]]) -> None:
    check(path, lambda df: df.filter(pz.col("a") > 1).select(pz.col("c").sum()))
    assert reads == [("a", "c")]


def test_scan_keeps_a_column_for_the_row_count(
    path: str, reads: list[tuple[str, ...]]
) -> None:
    assert check(path, lambda df: df.select(pz.len())) == {"len": [4]}
    assert reads == [("a",)]


@pytest.mark.parametrize(
    "query",
    [
        lambda df: df.with_columns(a=pz.col("c") * 2).select("a", "b"),
        lambda df: df.with_columns(e=pz.col("a") + 1).drop("d").sort("e"),
        lambda df: df.rename({"a": "x", "b": "a"}).select("x", "a"),
        lambda df: df.group_by("b").agg(pz.col("a").sum()).sort("b"),
        lambda df: df.group_by(pz.col("b").str.to_uppercase()).agg(pz.len()).sort("b"),
        lambda df: df.with_row_index("i", order_by="a").select("i", "c"),
        lambda df: df.unique(["b"], keep="any", order_by=["a"]).select("a").sort("a"),
        lambda df: df.select(pz.col("a").cum_sum().over(order_by="c")),
        lambda df: df.top_k(2, by="a").select("b"),
        lambda df: df.with_columns(pz.when(pz.col("d")).then(pz.col("a"))).select("a"),
    ],
)
def test_pruned_queries_match_eager(
    path: str, query: Callable[[Any], Any], reads: list[tuple[str, ...]]
) -> None:
    check(path, query)
    assert len(reads[0]) < len(DATA)


@pytest.mark.parametrize(
    "query",
    [
        lambda df: df.select(pz.all()),
        lambda df: df.select(pz.nth(2)),
        lambda df: df.select(pz.exclude("a")),
        lambda df: df.unique(keep="any").sort("a").select("a"),
        lambda df: df.unique([], keep="any").sort("a").select("a"),
        lambda df: df.drop_nulls([]).select("a"),
    ],
)
def test_unknown_columns_read_everything(
    path: str, query: Callable[[Any], Any], reads: list[tuple[str, ...]]
) -> None:
    check(path, query)
    assert reads == [tuple(DATA)]


def test_selectors_need_the_schema(path: str, reads: list[tuple[str, ...]]) -> None:
    # Dtypes are inferred from values, so resolving a selector collects the input once.
    check(path, lambda df: df.select(pz.selectors.numeric()))
    assert reads[0] == tuple(DATA)


def test_join_keeps_suffixed_collisions(path: str) -> None:
    other = {"a": [1, 2], "b": ["left", "right"], "e": [0, 1]}

    def query(df: Any) -> Any:
        right = pz.from_dict(other)
        right = right.lazy() if isinstance(df, pz.LazyFrame) else right
        return df.join(right, on="a").select("a", "b_right")

    assert check(path, query) == {"a": [1, 2], "b_right": ["left", "right"]}


@pytest.mark.parametrize("how", ["inner", "left", "full", "semi", "anti", "cross"])
def test_joins_match_eager(path: str, how: str) -> None:
    def query(df: Any) -> Any:
        other = df.select(pz.col("a"), pz.col("c").alias("e"))
        kwargs = {} if how == "cross" else {"on": "a"}
        return df.join(other, how=how, **kwargs)

    check(path, query, ordered=False)


def test_shared_subplan_provides_what_each_use_needs(
    path: str, reads: list[tuple[str, ...]]
) -> None:
    def query(df: Any) -> Any:
        left = df.select("a", "b")
        right = df.select("a", "c")
        return left.join(right, on="a")

    check(path, query, ordered=False)
    assert reads == [("a", "b", "c")]  # one scan, shared by both sides


def test_concat(path: str, reads: list[tuple[str, ...]]) -> None:
    def query(df: Any) -> Any:
        return pz.concat([df, df]).select(pz.col("a").sum())

    assert check(path, query) == {"a": [20]}
    assert reads == [("a",)]


def test_vertical_concat_of_pruned_and_computed_inputs(path: str) -> None:
    computed = pz.scan_parquet(path).select(pz.col("a"), pz.col("b"))
    lf = pz.concat([pz.scan_parquet(path, columns=["a", "b"]), computed]).select("a")
    assert lf.collect().to_native() == {"a": [*DATA["a"], *DATA["a"]]}


def test_dtype_dependent_output_names(path: str) -> None:
    def query(df: Any) -> Any:
        return df.with_columns(pz.selectors.numeric().name.suffix("_n"))

    lf = query(pz.scan_parquet(path))
    assert lf.columns == [*DATA, "a_n", "c_n"]
    check(path, query)

    other = pz.from_dict({"a": [1, 2], "x": [0, 1]})

    def join(df: Any) -> Any:
        right = other.lazy() if isinstance(df, pz.LazyFrame) else other
        return df.join(query(right), on="a").select("a", "x_n")

    check(path, join, ordered=False)


def test_errors_are_raised_when_operations_are_defined(path: str) -> None:
    lf = pz.scan_parquet(path)
    with pytest.raises(ColumnNotFoundError):
        lf.select("missing")
    with pytest.raises(ColumnNotFoundError):
        lf.filter(pz.col("missing") > 1)


def test_scan_parquet_columns_and_file_like(path: str) -> None:
    with open(path, "rb") as stream:  # noqa: PTH123
        lf = pz.scan_parquet(BytesIO(stream.read()), columns=["c", "a"])
    assert lf.collect_schema().names() == ["c", "a"]
    assert lf.collect().to_native() == {"c": DATA["c"], "a": DATA["a"]}
    with pytest.raises(ValueError, match="not found"):
        pz.scan_parquet(path, columns=["missing"])
    with pytest.raises(TypeError, match="Unsupported keyword"):
        pz.scan_parquet(path, engine="pyarrow")


def test_columns_without_reading(path: str, reads: list[tuple[str, ...]]) -> None:
    lf = pz.scan_parquet(path).with_columns(e=pz.col("a") * 2).drop("b")
    assert lf.columns == ["a", "c", "d", "e"]
    assert reads == []


def test_explain(path: str) -> None:
    lf = pz.scan_parquet(path).filter(pz.col("a") > 1).select("b")
    assert lf.to_native().explain().splitlines() == [
        "SIMPLE_SELECT ['b']",
        "  FILTER",
        f"    SCAN PARQUET {path!r} ['a', 'b', 'c', 'd']",
    ]
    assert "FRAME ['a', 'b', 'c', 'd']" in repr(pz.from_dict(DATA).lazy())


def test_drop_then_len() -> None:
    lf = pz.from_dict({"a": [1, 2], "b": [3, 4]}).lazy()
    assert lf.drop("a").select(pz.len()).collect().to_native() == {"len": [2]}
