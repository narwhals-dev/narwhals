"""Opt-in thread parallelism for the hot per-row loops.

Set ``NARWHALS_DICT_THREADS=<n>`` (``n >= 2``) to spread the large columnar
kernels (elementwise maps, filter masks, gathers, join probes) across ``n``
threads. On a regular CPython build the GIL serializes pure-Python loops, so
this only pays off on a free-threaded build (``python3.14t``); it is off by
default and everything stays on the existing serial paths.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

T = TypeVar("T")

THREADS: int = int(os.environ.get("NARWHALS_DICT_THREADS", "0") or "0")
"""Number of worker threads; below 2 every kernel stays serial."""

MIN_ROWS: int = int(os.environ.get("NARWHALS_DICT_PARALLEL_MIN_ROWS", "10000"))
"""Rows below which threading overhead outweighs any possible win.

Overridable via ``NARWHALS_DICT_PARALLEL_MIN_ROWS``, mainly so the test suite
(tiny frames) can force the parallel paths on.
"""

_pool: ThreadPoolExecutor | None = None


def should_parallelize(n_rows: int) -> bool:
    return THREADS >= 2 and n_rows >= MIN_ROWS


def _get_pool() -> ThreadPoolExecutor:
    global _pool  # noqa: PLW0603
    if _pool is None:
        _pool = ThreadPoolExecutor(
            max_workers=THREADS, thread_name_prefix="narwhals-dict"
        )
    return _pool


def map_tasks(fn: Callable[..., T], items: Iterable[Any]) -> list[T]:
    """Apply `fn` to every item on the pool, preserving order.

    Must not be called from inside another parallel task: the pool is shared,
    and a task waiting on the pool it runs on can deadlock.
    """
    return list(_get_pool().map(fn, items))


def chunk_bounds(n_rows: int) -> list[tuple[int, int]]:
    """Split `range(n_rows)` into contiguous `(start, stop)` chunks.

    Over-decomposes into ``4 * THREADS`` chunks so slower cores (e.g. the
    efficiency cores on heterogeneous CPUs) grab fewer chunks instead of
    stalling the whole map on one oversized chunk.
    """
    size = max(-(-n_rows // (THREADS * 4)), MIN_ROWS // 4, 1)  # ceil
    return [(start, min(start + size, n_rows)) for start in range(0, n_rows, size)]


def run_chunks(kernel: Callable[[int, int], T], n_rows: int) -> list[T]:
    """Run `kernel(start, stop)` per chunk on the pool, results in chunk order."""
    return list(_get_pool().map(lambda bounds: kernel(*bounds), chunk_bounds(n_rows)))


def gather_chunks(kernel: Callable[[int, int], list[T]], n_rows: int) -> list[T]:
    """Concatenate the per-chunk lists of `run_chunks` back into one column."""
    result: list[T] = []
    for part in run_chunks(kernel, n_rows):
        result.extend(part)
    return result
