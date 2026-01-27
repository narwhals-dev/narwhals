from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

    from tpch.classes import Backend, Query


def test_execute_query(
    query: Query, backend: Backend, scale_factor: float, request: pytest.FixtureRequest
) -> None:
    """Helper function to run a TPCH query test."""
    result = query.try_run(backend, scale_factor)
    query.assert_expected(result, backend, scale_factor, request)
