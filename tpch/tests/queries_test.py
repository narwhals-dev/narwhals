from __future__ import annotations

from typing import TYPE_CHECKING

from polars.testing import assert_frame_equal

from narwhals.exceptions import NarwhalsError

if TYPE_CHECKING:
    from tpch.tests.conftest import Backend, Query


def test_execute_query(query: Query, backend: Backend, scale_factor: float) -> None:
    """Helper function to run a TPCH query test."""
    try:
        result = query.run(backend)
    except NarwhalsError as exc:
        msg = f"Query [{query}-{backend}] ({scale_factor=}) failed with the following error in Narwhals:\n{exc}"
        raise RuntimeError(msg) from exc
    try:
        assert_frame_equal(query.expected(), result, check_dtypes=False)
    except AssertionError as exc:
        msg = f"Query [{query}-{backend}] ({scale_factor=}) resulted in wrong answer:\n{exc}"
        raise AssertionError(msg) from exc
