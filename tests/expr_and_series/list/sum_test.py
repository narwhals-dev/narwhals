from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import DUCKDB_VERSION, PANDAS_VERSION, assert_equal_data

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager

data = {"a": [[3, None, 2, 2, 4, None], [-1], None, [None, None, None], []]}
expected = [11, -1, None, 0, 0]


def test_sum_expr(
    request: pytest.FixtureRequest, nw_frame_constructor: Constructor
) -> None:
    if any(backend in str(nw_frame_constructor) for backend in ("dask", "cudf")):
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(nw_frame_constructor):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")
    if "duckdb" in str(nw_frame_constructor) and DUCKDB_VERSION < (1, 2):
        reason = "version too old, duckdb 1.2 required for LambdaExpression."
        pytest.skip(reason=reason)
    result = nw.from_native(nw_frame_constructor(data)).select(
        nw.col("a").cast(nw.List(nw.Int32())).list.sum()
    )
    assert_equal_data(result, {"a": expected})


def test_sum_series(
    request: pytest.FixtureRequest, nw_eager_constructor: ConstructorEager
) -> None:
    if any(backend in str(nw_eager_constructor) for backend in ("cudf",)):
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(nw_eager_constructor):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")
    df = nw.from_native(nw_eager_constructor(data), eager_only=True)
    result = df["a"].cast(nw.List(nw.Int32())).list.sum()
    assert_equal_data({"a": result}, {"a": expected})
