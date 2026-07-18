"""Concurrency stress tests for narwhals-owned shared state.

Targets: caches populated on first use, lazily-materialized dtype metadata,
expression `over` push-down, and `narwhals.sql`'s shared DuckDB catalog.

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
from tests.utils import DUCKDB_VERSION

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future


def run_threaded(
    func: Callable[..., None],
    max_workers: int = 8,
    *,
    pass_count: bool = False,
    pass_barrier: bool = False,
    outer_iterations: int = 1,
    prepare_args: Callable[[], list[Any]] | None = None,
) -> None:
    """Runs a function many times in parallel.

    Ported from NumPy's `run_threaded` test helper, the pattern recommended by
    https://py-free-threading.github.io/testing/. Only type hints and
    keyword-only booleans were added.
    Source: https://github.com/numpy/numpy/blob/7e1f94485495485c5cc3a408ab9e945940f1b91f/numpy/testing/_private/utils.py#L2831
    Copyright (c) 2005-2025, NumPy Developers. License: BSD 3-Clause.
    """
    for _ in range(outer_iterations):
        with ThreadPoolExecutor(max_workers=max_workers) as tpe:
            args = [] if prepare_args is None else prepare_args()
            barrier = threading.Barrier(max_workers) if pass_barrier else None
            if barrier is not None:
                args.append(barrier)
            if pass_count:
                all_args = [(i, *args) for i in range(max_workers)]
            else:
                all_args = [tuple(args) for _ in range(max_workers)]
            futures: list[Future[None]] = []
            try:
                futures.extend(tpe.submit(func, *arg) for arg in all_args)
            except RuntimeError as e:  # pragma: no cover
                pytest.skip(
                    f"Spawning {max_workers} threads failed with error {e!r} "
                    "(likely due to resource limits on the system running the tests)"
                )
            finally:
                if len(futures) < max_workers and barrier is not None:
                    barrier.abort()
            for f in futures:
                f.result()


def test_gil_stays_disabled_on_free_threaded_build() -> None:
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        pytest.skip("not a free-threaded build")
    is_gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)
    assert not is_gil_enabled(), (
        "The GIL was re-enabled, likely by importing an extension module "
        "without free-threading support."
    )


def test_from_native_cold_caches() -> None:
    pytest.importorskip("polars")
    import polars as pl

    from narwhals import _utils

    df = pl.DataFrame({"a": [1, 2, 3]})

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
        nw_df = nw.from_native(df, eager_only=True)
        assert nw_df["a"].sum() == 6

    run_threaded(check, pass_barrier=True, outer_iterations=3, prepare_args=clear_caches)


def test_plugin_discovery_cold_cache() -> None:
    from narwhals import plugins

    def clear_caches() -> list[Any]:
        plugins._discover_entrypoints.cache_clear()
        return []

    def check(barrier: threading.Barrier) -> None:
        barrier.wait()
        assert plugins._discover_entrypoints() is not None

    run_threaded(check, pass_barrier=True, outer_iterations=3, prepare_args=clear_caches)


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

    run_threaded(check, pass_barrier=True, outer_iterations=5)


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

    run_threaded(check, pass_barrier=True, outer_iterations=3, prepare_args=clear_caches)


def test_shared_lazyframe_schema() -> None:
    pytest.importorskip("duckdb")
    import duckdb

    rel = duckdb.connect().sql("select 1::BIGINT as a, 'x' as b")
    lf = nw.from_native(rel)
    expected = {"a": nw.Int64(), "b": nw.String()}

    def check(barrier: threading.Barrier) -> None:
        barrier.wait()
        assert lf.collect_schema() == expected
        assert lf.columns == ["a", "b"]

    run_threaded(check, pass_barrier=True, outer_iterations=5)


def test_sql_table_concurrent() -> None:
    pytest.importorskip("duckdb")
    if DUCKDB_VERSION < (1, 3):
        pytest.skip()
    from narwhals.sql import table

    def check(barrier: threading.Barrier) -> None:
        barrier.wait()
        name = f"tbl_{uuid.uuid4().hex}"
        result = table(name, {"a": nw.Int64(), "b": nw.String()})
        assert result.collect_schema() == {"a": nw.Int64(), "b": nw.String()}
        assert name in result.to_sql()

    run_threaded(check, pass_barrier=True, outer_iterations=5)
