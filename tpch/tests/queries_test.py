from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import pytest

from tpch.execute import BACKEND_NAMESPACE_KWARGS_MAP
from tpch.execute import DUCKDB_SKIPS
from tpch.execute import _execute_query_single_backend
from tpch.execute import execute_query

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_codspeed.plugin import BenchmarkFixture


ROOT_PATH = Path(__file__).resolve().parent.parent
# Directory containing all the query scripts
QUERIES_DIR = ROOT_PATH / "queries"


@pytest.mark.parametrize("query_path", QUERIES_DIR.glob("q[1-9]*.py"))
def test_execute_query(query_path: Path) -> None:
    print(f"executing query {query_path.stem}")  # noqa: T201
    _ = execute_query(query_id=query_path.stem)


@pytest.mark.parametrize("query_path", QUERIES_DIR.glob("q[1-9]*.py"))
@pytest.mark.parametrize(
    ("backend", "namespace_and_kwargs"), BACKEND_NAMESPACE_KWARGS_MAP.items()
)
def test_benchmark_query(
    benchmark: BenchmarkFixture,
    query_path: Path,
    backend: str,
    namespace_and_kwargs: tuple[ModuleType, dict[str, Any]],
) -> None:
    query_id = query_path.stem
    native_namespace, kwargs = namespace_and_kwargs

    if (
        backend in {"duckdb", "sqlframe"} and query_id in DUCKDB_SKIPS
    ) or backend == "dask":
        pytest.skip()

    _ = benchmark(
        lambda: _execute_query_single_backend(
            query_id=query_id, native_namespace=native_namespace, **kwargs
        )
    )
