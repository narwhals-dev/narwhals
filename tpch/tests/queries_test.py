from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

    from tpch.classes import Backend, Query


def test_execute_query(
    query: Query, backend: Backend, request: pytest.FixtureRequest
) -> None:
    """Helper function to run a TPCH query test."""
    query.execute(backend, request)
