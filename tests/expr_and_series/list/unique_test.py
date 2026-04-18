from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import DUCKDB_VERSION

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager

data = {"a": [[2, 2, 3, None, None], None, [], [None]]}
expected = {2, 3, None}


def test_unique_expr(
    request: pytest.FixtureRequest, nw_frame_constructor: Constructor
) -> None:
    if any(
        backend in str(nw_frame_constructor)
        for backend in ("dask", "modin", "cudf", "pyarrow", "pandas")
    ):
        request.applymarker(pytest.mark.xfail)
    if "duckdb" in str(nw_frame_constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    result = (
        nw.from_native(nw_frame_constructor(data))
        .select(nw.col("a").cast(nw.List(nw.Int32())).list.unique())
        .lazy()
        .collect()["a"]
        .to_list()
    )
    # We don't yet have `.list.sort` to get deterministic order, and pyarrow
    # doesn't support `explode`, so we can't guarantee the order of elements.
    # However, we can check that the unique values are present.
    assert len(result) == 4
    assert len(result[0]) == 3
    assert set(result[0]) == {2, 3, None}
    assert result[1] is None
    assert len(result[2]) == 0
    assert len(result[3]) == 1


def test_unique_series(
    request: pytest.FixtureRequest, nw_eager_constructor: ConstructorEager
) -> None:
    if any(
        backend in str(nw_eager_constructor)
        for backend in ("modin", "cudf", "pyarrow", "pandas")
    ):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(nw_eager_constructor(data), eager_only=True)
    result = df["a"].cast(nw.List(nw.Int32())).list.unique().to_list()
    assert len(result) == 4
    assert len(result[0]) == 3
    assert set(result[0]) == {2, 3, None}
    assert result[1] is None
    assert len(result[2]) == 0
    assert len(result[3]) == 1
