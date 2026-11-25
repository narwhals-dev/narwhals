from __future__ import annotations

from io import BytesIO, StringIO
from typing import TYPE_CHECKING, Any

import pizzlies as pz
import pytest
from pizzlies._plugin import lazy
from pizzlies._plugin.lazy import FramePlan, LazyPlan

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from narwhals.exceptions import ColumnNotFoundError, DuplicateError

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
        assert list(result.items()) == list(expected.items())
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
        lambda df: df.with_columns(e=pz.col("a") + 1).sort("e").select("e"),
        lambda df: df.drop_nulls([]).select("a"),
        lambda df: df.rename({"a": "x", "b": "a"}).select("x", "a"),
        lambda df: df.group_by("b").agg(pz.col("a").sum()).sort("b"),
        lambda df: df.group_by(pz.col("b").str.to_uppercase()).agg(pz.len()).sort("b"),
        lambda df: df.with_row_index("i", order_by="a").select("i", "c"),
        lambda df: df.unique(["b"], keep="any", order_by=["a"]).select("a").sort("a"),
        lambda df: df.select(pz.col("a").cum_sum().over(order_by="c")),
        lambda df: df.top_k(2, by="a").select("b"),
        lambda df: df.with_columns(pz.when(pz.col("d")).then(pz.col("a"))).select("a"),
        lambda df: df.with_columns(x=pz.lit(1)).select("a", "x"),
        lambda df: df.with_columns(s=pz.col("a").sum()).select("s"),
        lambda df: df.select(pz.col("a"), pz.col("c").max()),
        lambda df: df.select(pz.col("d").all()),
        lambda df: df.drop("d").select("a"),
        lambda df: df.drop("d"),
        lambda df: df.with_columns(e=pz.col("a") * 2).drop("b", "c", "d"),
        lambda df: df.unpivot(on=["a"], index=["b"]),
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

    def diagonal(df: Any) -> Any:
        return pz.concat([df, df.select("a")], how="diagonal").select("a")

    check(path, diagonal)


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


def test_errors_are_raised_on_collect(path: str, reads: list[tuple[str, ...]]) -> None:
    lf = pz.scan_parquet(path)
    # The hint lists every column, not just those a pruned plan would read.
    hint = r"Did you mean one of these columns: \['a', 'b', 'c', 'd'\]"
    with pytest.raises(ColumnNotFoundError, match=hint):
        lf.select("missing").collect()
    with pytest.raises(ColumnNotFoundError, match=hint):
        lf.filter(pz.col("missing") > 1).select("a").collect()
    with pytest.raises(ColumnNotFoundError, match=hint):
        lf.sort("missing").collect()
    with pytest.raises(ColumnNotFoundError):
        lf.join(lf, left_on="a", right_on="missing").collect()
    assert reads == []  # raised before reading anything


def test_user_errors_run_the_plan_once(
    path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    original = lazy._execute

    def spy(plan: LazyPlan, context: Any) -> Any:
        calls.append(plan._describe())
        return original(plan, context)

    monkeypatch.setattr(lazy, "_execute", spy)
    lf = pz.scan_parquet(path).filter(pz.col("a").str.len_chars() > 0).select("a")
    with pytest.raises(Exception):  # noqa: B017, PT011
        lf.collect()
    assert len(calls) == 1


def test_schema_runs_the_plan_once(path: str, reads: list[tuple[str, ...]]) -> None:
    lf = pz.scan_parquet(path).select("a", "b")
    assert lf.columns == ["a", "b"]
    assert lf.collect_schema() == {"a": pz.Int64(), "b": pz.String()}
    assert reads == [("a", "b")]


def test_columns_do_not_run(path: str, reads: list[tuple[str, ...]]) -> None:
    lf = pz.scan_parquet(path)
    assert lf.columns == list(DATA)
    assert pz.from_dict(DATA).lazy().columns == list(DATA)
    other = pz.from_dict({"a": [1], "b": [2]}).lazy()
    assert lf.with_columns(e=pz.col("a") * 2).drop("b").join(other, on="a").group_by(
        "a"
    ).agg(pz.all().sum()).columns == ["a", "c", "d", "e", "b"]
    assert reads == []
    # Narwhals checks that the columns exist in every `cast`.
    lf = pz.scan_parquet(path).cast({"a": pz.Float64}).select("a")
    assert lf.collect().to_native() == {"a": [1.0, 2.0, 3.0, 4.0]}
    assert reads == [("a",)]


def test_collect_results_are_independent() -> None:
    lf = pz.from_dict({"a": [1, 2]}).lazy().with_columns(b=pz.col("a") + 1)
    first = lf.collect().to_native()
    first["z"] = [0]
    assert lf.collect().to_native() == {"a": [1, 2], "b": [2, 3]}


def test_drop_keeps_the_row_count() -> None:
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    other = pz.from_dict({"y": [10, 20]})
    for frame, right in (
        (pz.from_dict(data), other),
        (pz.from_dict(data).lazy(), other.lazy()),
    ):
        query = frame.drop("a").join(right, how="cross").select(pz.col("y").sum())
        result = query.collect() if isinstance(query, pz.LazyFrame) else query
        assert result.to_native() == {"y": [90]}
        index = frame.drop("a").with_row_index("i", order_by=None).select("i")
        result = index.collect() if isinstance(index, pz.LazyFrame) else index
        assert result.to_native() == {"i": [0, 1, 2]}


@pytest.mark.parametrize(
    "query",
    [
        lambda df: df.rename({"a": "b"}).select("b"),
        lambda df: df.with_row_index("b", order_by="a").select("a"),
        lambda df: df.join(
            df.with_columns(b_right=pz.col("b")), on="a", suffix="_right"
        ).select("a"),
    ],
)
def test_pruning_keeps_duplicate_errors(path: str, query: Callable[[Any], Any]) -> None:
    with pytest.raises(DuplicateError):
        query(pz.read_parquet(path))
    with pytest.raises(DuplicateError):
        query(pz.scan_parquet(path)).collect()


def test_stable_v1_only_methods_are_not_implemented() -> None:
    lf = nw_v1.from_native(pz.from_dict(DATA).lazy().to_native())
    with pytest.raises(NotImplementedError):
        lf.tail(1)


def test_scan_parquet_columns_and_file_like(path: str) -> None:
    with open(path, "rb") as stream:  # noqa: PTH123
        lf = pz.scan_parquet(BytesIO(stream.read()), columns=["c", "a"])
    assert lf.collect_schema().names() == ["c", "a"]
    assert lf.collect().to_native() == {"c": DATA["c"], "a": DATA["a"]}
    with pytest.raises(ValueError, match="not found"):
        pz.scan_parquet(path, columns=["missing"])
    with pytest.raises(TypeError, match="Unsupported keyword"):
        pz.scan_parquet(path, engine="pyarrow")
    with pytest.raises(DuplicateError):
        pz.scan_parquet(path, columns=["a", "a"])
    with pytest.raises(TypeError, match="binary file-like"):
        pz.scan_parquet(StringIO("not parquet"))


def test_explain(path: str) -> None:
    lf = pz.scan_parquet(path).filter(pz.col("a") > 1).select("b")
    assert lf.to_native().explain().splitlines() == [
        "SIMPLE_SELECT ['b']",
        "  FILTER",
        f"    SCAN PARQUET {path!r} ['a', 'b', 'c', 'd']",
    ]
    assert "FRAME ['a', 'b', 'c', 'd']" in repr(pz.from_dict(DATA).lazy())
    assert "DROP [['a']]" in pz.from_dict(DATA).lazy().drop("a").to_native().explain()


def test_with_columns_overwrite_keeps_position(path: str) -> None:
    def query(df: Any) -> Any:
        return df.with_columns(a=pz.col("b") + "!")

    lf = query(pz.scan_parquet(path))
    assert lf.columns == list(DATA)
    assert lf.collect_schema().names() == list(DATA)
    assert list(check(path, query)) == list(DATA)


def test_drop_then_len() -> None:
    lf = pz.from_dict({"a": [1, 2], "b": [3, 4]}).lazy()
    assert lf.drop("a").select(pz.len()).collect().to_native() == {"len": [2]}


def test_scan_csv_is_lazy(tmp_path: Path) -> None:
    file = tmp_path / "data.csv"
    file.write_text("a,b\n1,x\n2,y\n")
    lf = pz.scan_csv(str(file))
    assert isinstance(lf.to_native(), FramePlan)
    assert lf.select("b").collect().to_native() == {"b": ["x", "y"]}
    file.write_text("a;b\n1;x\n")
    assert pz.scan_csv(str(file), separator=";").collect().to_native() == {
        "a": [1],
        "b": ["x"],
    }
    with pytest.raises(TypeError, match="Use `separator`"):
        pz.scan_csv(str(file), delimiter=";")


def _both(data: dict[str, list[Any]], query: Callable[[Any], Any]) -> list[Any]:
    """`query`'s result (or exception type) on an eager frame, and on a lazy one."""
    outcomes: list[Any] = []
    for frame in (pz.from_dict(data), pz.from_dict(data).lazy()):
        try:
            result = query(frame)
            if isinstance(result, pz.LazyFrame):
                result = result.collect()
            outcomes.append(result.to_native())
        except Exception as exc:  # noqa: BLE001, PERF203
            outcomes.append(type(exc))
    return outcomes


@pytest.mark.parametrize(
    ("data", "query", "expected"),
    [
        (
            {"a": [1], "b": [2]},
            lambda df: pz.concat(
                [df.select("a"), df.select(pz.col("b").alias("c"))], how="diagonal"
            ),
            {"a": [1, None], "c": [None, 2]},
        ),
        ({"a": [1]}, lambda df: df.select("a", "a"), DuplicateError),
        (
            {"c": [1, None], "d": [None, 2], "e": [1, 1]},
            lambda df: df.group_by("c", drop_null_keys=True).agg(
                pz.selectors.numeric().max()
            ),
            {"c": [1], "d": [None], "e": [1]},
        ),
        (
            {"k": [1, 2], "c": [True, False]},
            lambda df: df.group_by("k").agg(pz.selectors.numeric().max()).sort("k"),
            {"k": [1, 2]},
        ),
        (
            {"a": [1, 2], "b": [3, 4]},
            lambda df: df.group_by(
                (pz.col("a") * 2).alias("g"), (pz.col("b") * 2).alias("g")
            ).agg(),
            DuplicateError,
        ),
        (
            {"a": [1, 2]},
            lambda df: df.group_by(pz.col("a").alias("k")).agg(pz.col("k").sum()),
            ColumnNotFoundError,
        ),
        (
            {"a": [1, 2]},
            lambda df: df.group_by("zz").agg(pz.col("a").sum()),
            ColumnNotFoundError,
        ),
    ],
)
def test_eager_and_lazy_agree(
    data: dict[str, list[Any]], query: Callable[[Any], Any], expected: Any
) -> None:
    assert _both(data, query) == [expected, expected]


def test_long_plans_do_not_recurse() -> None:
    lf = pz.from_dict({"a": [1, 2]}).lazy()
    for _ in range(2_000):
        lf = lf.with_columns(pz.col("a") + 1)
    assert lf.collect().to_native() == {"a": [2_001, 2_002]}
    assert len(lf.to_native().explain().splitlines()) == 2_001


def test_shared_subplans_are_explained_and_run_once(
    path: str, reads: list[tuple[str, ...]]
) -> None:
    lf = pz.scan_parquet(path).select("a", "c")
    for _ in range(10):
        lf = pz.concat([lf, lf])
    explained = lf.to_native().explain().splitlines()
    assert len(explained) < 100
    assert "<see [1]>" in explained[-1]
    # Resolving the selector runs the input once, and so does `collect`.
    assert lf.select(pz.selectors.numeric()).columns == ["a", "c"]
    assert len(lf.select(pz.selectors.numeric()).collect()) == 4 * 2**10
    assert reads == [("a", "c"), ("a", "c")]
