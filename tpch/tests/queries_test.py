from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tpch.execute import execute_query

if TYPE_CHECKING:
    from pytest_codspeed.plugin import BenchmarkFixture


ROOT_PATH = Path(__file__).resolve().parent.parent
# Directory containing all the query scripts
QUERIES_DIR = ROOT_PATH / "queries"


@pytest.mark.parametrize("query_path", QUERIES_DIR.glob("q[1-9]*.py"))
def test_execute_query(query_path: Path) -> None:
    print(f"executing query {query_path.stem}")  # noqa: T201
    _ = execute_query(query_id=query_path.stem)


@pytest.mark.parametrize("query_path", QUERIES_DIR.glob("q[1-9]*.py"))
def test_benchmark_query(benchmark: BenchmarkFixture, query_path: Path) -> None:
    print(f"executing query {query_path.stem}")  # noqa: T201
    _ = execute_query(query_id=query_path.stem, benchmark=benchmark)
