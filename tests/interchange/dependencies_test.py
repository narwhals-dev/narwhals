from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import pytest

import narwhals.stable.v1 as nw_v1
from narwhals.stable.v1.dependencies import (
    is_into_dataframe as v1_is_into_dataframe,
    is_into_lazyframe as v1_is_into_lazyframe,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import duckdb
    import ibis

    Interchange: TypeAlias = duckdb.DuckDBPyRelation | ibis.Table


@pytest.fixture
def data() -> dict[str, Any]:
    return {"a": [1, 2, 3], "b": [4, 5, 6]}


@pytest.fixture
def frame(constructor: Callable[[Any], Interchange], data: dict[str, Any]) -> Interchange:
    name = str(constructor)
    if "duckdb" in name or "ibis" in name:
        return constructor(data)
    pytest.skip("non-interchange frames are checked in other tests")


def test_is_into_dataframe(frame: Interchange) -> None:
    nw_v1_frame_1 = nw_v1.from_native(frame, eager_or_interchange_only=True)
    assert v1_is_into_dataframe(nw_v1_frame_1)
    nw_v1_frame_2 = nw_v1.from_native(frame)
    assert v1_is_into_dataframe(nw_v1_frame_2)


def test_is_into_lazyframe(frame: Interchange) -> None:
    nw_v1_frame_1 = nw_v1.from_native(frame, eager_or_interchange_only=True)
    assert v1_is_into_lazyframe(nw_v1_frame_1) is False
    nw_v1_frame_2 = nw_v1.from_native(frame)
    assert v1_is_into_lazyframe(nw_v1_frame_2) is False


@pytest.mark.xfail(
    reason="https://github.com/narwhals-dev/narwhals/pull/3613#discussion_r3288440039"
)
def test_is_into_dataframe_native(frame: Interchange) -> None:
    assert v1_is_into_dataframe(frame)


@pytest.mark.xfail(
    reason="https://github.com/narwhals-dev/narwhals/pull/3613#discussion_r3288440039"
)
def test_is_into_lazyframe_native(frame: Interchange) -> None:
    assert v1_is_into_lazyframe(frame) is False
