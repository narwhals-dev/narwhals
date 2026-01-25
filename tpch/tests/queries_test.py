from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from polars.testing import assert_frame_equal

import narwhals as nw
from narwhals.exceptions import NarwhalsError
from tpch.tests.conftest import skip_if_unsupported

if TYPE_CHECKING:
    import polars as pl

    from narwhals.typing import IntoFrame

    DataLoader = Callable[[str], tuple[IntoFrame, ...]]

ROOT_PATH = Path(__file__).resolve().parent.parent
# Directory containing all the query scripts
QUERIES_DIR = ROOT_PATH / "queries"


def test_execute_query(
    query_id: str,
    backend_name: str,
    data_loader: DataLoader,
    expected_result: Callable[[str], pl.DataFrame],
) -> None:
    """Helper function to run a TPCH query test."""
    query_module = import_module(f"tpch.queries.{query_id}")

    skip_if_unsupported(query_id, backend_name)

    expected = expected_result(query_id)
    data = data_loader(query_id)

    try:
        result: pl.DataFrame = (
            query_module.query(*data)  # type: ignore[union-attr]
            .lazy()
            .collect(backend=nw.Implementation.POLARS)
            .to_native()
        )
    except NarwhalsError as exc:
        msg = f"Query {query_id} with {backend_name=} failed with the following error in Narwhals:\n{exc}"
        raise RuntimeError(msg) from exc

    try:
        assert_frame_equal(expected, result, check_dtypes=False)
    except AssertionError as exc:
        msg = f"Query {query_id} with {backend_name=} resulted in wrong answer:\n{exc}"
        raise AssertionError(msg) from exc
