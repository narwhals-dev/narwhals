"""Concurrency stress tests for narwhals-owned shared state.

Targets: caches populated on first use, lazily-materialized dtype metadata,
expression `over` push-down, state stashed on shared Narwhals objects, and
`narwhals.sql`'s shared DuckDB catalog.

See [docs/concepts/thread_safety.md].

NOTE: The tests are valid on any build (races are bugs under the GIL too), but are
most effective on a free-threaded build (`PYTHON_GIL=0`), where threads run in parallel.
"""

from __future__ import annotations

import sys
import sysconfig
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from tests.utils import PYARROW_VERSION, assert_equal_data

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future

    from tests.utils import ConstructorEager

DATA: dict[str, Any] = {
    "g": [1, 1, 2, 2, 3, 3],
    "i": [6, 5, 4, 3, 2, 1],
    "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
}
"""`i` is a unique ordering column, reversing the row order."""

_SELECT_TZ = "select timestamptz '2024-01-01' as t"
"""Time-zone-aware: forces Narwhals to query the connection for its time zone."""


def run_threaded(
    func: Callable[..., None],
    max_workers: int = 8,
    *,
    outer_iterations: int = 1,
    prepare_args: Callable[[], list[Any]] | None = None,
) -> None:
    """Run `func` in `max_workers` threads at once, `outer_iterations` times.

    Each thread receives a shared `threading.Barrier` as its final positional argument,
    so callers can line every thread up on the racy window with `barrier.wait()` before proceeding.
    `prepare_args`, when given, is called once per iteration to build the arguments that precede
    the barrier (e.g. to reset caches).

    Adapted from NumPy's `run_threaded` test helper, the pattern recommended by
    https://py-free-threading.github.io/testing/.
    Source: https://github.com/numpy/numpy/blob/7e1f94485495485c5cc3a408ab9e945940f1b91f/numpy/testing/_private/utils.py#L2831
    Copyright (c) 2005-2025, NumPy Developers. License: BSD 3-Clause.
    """
    for _ in range(outer_iterations):
        with ThreadPoolExecutor(max_workers=max_workers) as tpe:
            args = [] if prepare_args is None else prepare_args()
            barrier = threading.Barrier(max_workers)
            args.append(barrier)
            futures: list[Future[None]] = []
            try:
                futures.extend(tpe.submit(func, *args) for _ in range(max_workers))
            except RuntimeError as e:  # pragma: no cover
                # Release any threads already blocked on the barrier so the pool can
                # shut down instead of deadlocking, then skip.
                barrier.abort()
                pytest.skip(
                    f"Spawning {max_workers} threads failed with error {e!r} "
                    "(likely due to resource limits on the system running the tests)"
                )
            for f in futures:
                f.result()


def test_gil_stays_disabled_on_free_threaded_build() -> None:  # pragma: no cover
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        pytest.skip("not a free-threaded build")
    # NOTE: Only reached on a free-threaded build, never on the GIL-enabled coverage jobs.
    # Because of this, the entire function is flagged as "pragma: no cover"
    is_gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)
    assert not is_gil_enabled(), (
        "The GIL was re-enabled, likely by importing an extension module "
        "without free-threading support."
    )


def test_from_native_cold_caches() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    from narwhals import _utils

    tbl = pa.table({"a": [1, 2, 3]})

    def clear_caches() -> list[Any]:
        _utils.backend_version.cache_clear()
        _utils._import_native_namespace.cache_clear()
        _utils._version_namespace.cache_clear()
        _utils._version_dtypes.cache_clear()
        _utils._version_dataframe.cache_clear()
        _utils._version_lazyframe.cache_clear()
        _utils._version_series.cache_clear()
        return []

    def check(barrier: threading.Barrier) -> None:
        barrier.wait()
        nw_df = nw.from_native(tbl, eager_only=True)
        assert nw_df["a"].sum() == 6

    run_threaded(check, outer_iterations=3, prepare_args=clear_caches)


def test_plugin_discovery_cold_cache() -> None:
    from narwhals import plugins

    def clear_caches() -> list[Any]:
        plugins._discover_entrypoints.cache_clear()
        return []

    def check(barrier: threading.Barrier) -> None:
        barrier.wait()
        assert plugins._discover_entrypoints() is not None

    run_threaded(check, outer_iterations=3, prepare_args=clear_caches)


def test_shared_expr_over_push_down() -> None:
    # `.over()` must never mutate nodes reachable from the original,
    # potentially shared, expression.
    def make_expr() -> nw.Expr:
        return nw.col("a").cum_sum() + nw.col("b").cum_sum().abs()

    base = make_expr()
    expected_base = repr(base)
    expected_over = repr(make_expr().over("g", order_by="i"))

    def check(barrier: threading.Barrier) -> None:
        barrier.wait()
        result = base.over("g", order_by="i")
        assert repr(result) == expected_over
        assert repr(base) == expected_base

    run_threaded(check, outer_iterations=5)


def test_shared_expr_evaluation(constructor_eager: ConstructorEager) -> None:
    """Evaluating one shared `Expr` in several contexts at once must not mutate it."""
    df = nw.from_native(constructor_eager(DATA), eager_only=True)
    expr = nw.col("v").cum_sum()
    expected_repr = repr(expr)

    def check(barrier: threading.Barrier) -> None:
        barrier.wait()
        for _ in range(10):
            # Row order, `i` order, and one appended node: three rewrites of `expr`.
            assert_equal_data(df.select(expr), {"v": [1.0, 3.0, 6.0, 10.0, 15.0, 21.0]})
            assert_equal_data(
                df.select(expr.over(order_by="i")),
                {"v": [21.0, 20.0, 18.0, 15.0, 11.0, 6.0]},
            )
            assert_equal_data(
                df.with_columns(out=expr.abs()).select("out"),
                {"out": [1.0, 3.0, 6.0, 10.0, 15.0, 21.0]},
            )
            assert repr(expr) == expected_repr

    run_threaded(check, outer_iterations=2)


def test_shared_group_by_agg(constructor_eager: ConstructorEager) -> None:
    """A `GroupBy` reused from several threads must not leak state between them.

    Regression test: `agg` used to stash the native `groupby` on `self`, so one thread
    could read another's grouping and silently aggregate the wrong rows.
    """
    if "pyarrow_table" in str(constructor_eager) and PYARROW_VERSION < (14, 0):
        pytest.skip("https://github.com/apache/arrow/issues/36709")

    grouped = nw.from_native(constructor_eager(DATA), eager_only=True).group_by("g")
    # `sum` groups as-is, `first(order_by="i")` groups a sorted copy: two distinct
    # native groupings, so a leak between threads shows up in the values.
    unordered = {"g": [1, 2, 3], "v": [3.0, 7.0, 11.0]}
    ordered = {"g": [1, 2, 3], "v": [2.0, 4.0, 6.0]}

    def check(barrier: threading.Barrier) -> None:
        barrier.wait()
        for _ in range(20):
            res_ordered = grouped.agg(nw.col("v").first(order_by="i")).sort("g")
            assert_equal_data(res_ordered, ordered)

            res_unordered = grouped.agg(nw.col("v").sum()).sort("g")
            assert_equal_data(res_unordered, unordered)

    run_threaded(check, outer_iterations=3)


def test_shared_dataframe_read_only(constructor_eager: ConstructorEager) -> None:
    """Reading from a shared `DataFrame` is safe: no method may mutate it, or its input."""
    native = constructor_eager(DATA)
    df = nw.from_native(native, eager_only=True)
    native_before = repr(native)

    def check(barrier: threading.Barrier) -> None:
        barrier.wait()
        for _ in range(10):
            assert df.columns == ["g", "i", "v"]
            assert df.schema == {"g": nw.Int64(), "i": nw.Int64(), "v": nw.Float64()}
            assert df.shape == (6, 3)
            assert_equal_data(
                df.select(nw.col("v") * 2), {"v": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]}
            )
            assert_equal_data(
                df.filter(nw.col("g") == 1), {"g": [1, 1], "i": [6, 5], "v": [1.0, 2.0]}
            )
            assert_equal_data(
                df.sort("i").select("v"), {"v": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]}
            )
            assert_equal_data(df.unique("g").sort("g").select("g"), {"g": [1, 2, 3]})
            assert_equal_data(df.lazy().collect().select("g"), {"g": [1, 1, 2, 2, 3, 3]})
            # `scatter` is the one method that reads as in-place: it must not be.
            assert df["v"].scatter(0, 99.0).to_list() == [99.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            assert df["v"].to_list() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    run_threaded(check, outer_iterations=2)
    assert repr(native) == native_before, "narwhals mutated the native input"


def test_enum_deferred_categories() -> None:
    pytest.importorskip("polars")
    import polars as pl

    from narwhals._polars.utils import native_to_narwhals_dtype

    categories = ("ft_x", "ft_y", "ft_z")
    df = pl.DataFrame({"a": pl.Series(["ft_x"], dtype=pl.Enum(categories))})

    def clear_caches() -> list[Any]:
        native_to_narwhals_dtype.cache_clear()
        return []

    def check(barrier: threading.Barrier) -> None:
        barrier.wait()
        dtype = nw.from_native(df, eager_only=True).schema["a"]
        assert isinstance(dtype, nw.Enum)
        assert dtype.categories == categories

    run_threaded(check, outer_iterations=3, prepare_args=clear_caches)


def test_shared_lazyframe_schema() -> None:
    pytest.importorskip("duckdb")
    import duckdb

    con = duckdb.connect()
    rel = con.sql("select 1::BIGINT as a, 'x' as b")
    lf = nw.from_native(rel)
    expected = {"a": nw.Int64(), "b": nw.String()}

    def check(barrier: threading.Barrier) -> None:
        barrier.wait()
        assert lf.collect_schema() == expected
        assert lf.columns == ["a", "b"]

    run_threaded(check, outer_iterations=5)


def test_shared_duckdb_connection_schema() -> None:
    pytest.importorskip("duckdb")
    import duckdb

    con = duckdb.connect()
    con.sql("set timezone = 'UTC'")
    # One frame per thread *and* one shared frame, all on the same connection.
    frames = [nw.from_native(con.sql(_SELECT_TZ)) for _ in range(8)]
    shared = nw.from_native(con.sql(_SELECT_TZ))
    expected = {"t": nw.Datetime(time_zone="UTC")}

    def check(barrier: threading.Barrier) -> None:
        barrier.wait()
        for lf in (*frames, shared):
            assert lf.collect_schema() == expected

    run_threaded(check, outer_iterations=3)


def test_duckdb_per_thread_cursor() -> None:
    """The supported recipe for concurrent DuckDB use: one cursor per thread.

    NOTE: `TimeZone` is `LOCAL`-scoped, so a cursor does not inherit the parent's value.
    """
    pytest.importorskip("duckdb")
    import duckdb

    con = duckdb.connect()

    def check(barrier: threading.Barrier) -> None:
        cursor = con.cursor()
        cursor.sql("set timezone = 'UTC'")
        barrier.wait()
        for _ in range(5):
            lf = nw.from_native(cursor.sql(f"{_SELECT_TZ}, 1 as idx, 2 as a, 3 as b"))
            assert lf.collect_schema()["t"] == nw.Datetime(time_zone="UTC")
            assert_equal_data(lf.select("a").collect(), {"a": [2]})
            assert_equal_data(
                lf.unpivot(on=["a", "b"], index=["idx"]).sort("variable"),
                {"idx": [1, 1], "variable": ["a", "b"], "value": [2, 3]},
            )

    run_threaded(check, outer_iterations=3)


def test_sql_table_concurrent() -> None:
    pytest.importorskip("duckdb", minversion="1.3.0")
    from narwhals.sql import table

    def check(barrier: threading.Barrier) -> None:
        barrier.wait()
        name = f"tbl_{uuid.uuid4().hex}"
        result = table(name, {"a": nw.Int64(), "b": nw.String()})
        assert result.collect_schema() == {"a": nw.Int64(), "b": nw.String()}
        assert name in result.to_sql()

    run_threaded(check, outer_iterations=5)
