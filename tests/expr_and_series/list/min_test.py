from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, assert_equal_data

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager

data = {"a": [[3, None, 2, 2, 4, None], [-1], None, [None, None, None], []]}
expected = [2, -1, None, None, None]


def test_min_expr(
    request: pytest.FixtureRequest, nw_frame_constructor: Constructor
) -> None:
    if any(backend in str(nw_frame_constructor) for backend in ("dask", "cudf")):
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(nw_frame_constructor):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")
    result = nw.from_native(nw_frame_constructor(data)).select(
        nw.col("a").cast(nw.List(nw.Int32())).list.min()
    )
    assert_equal_data(result, {"a": expected})


def test_min_series(
    request: pytest.FixtureRequest, nw_eager_constructor: ConstructorEager
) -> None:
    if any(backend in str(nw_eager_constructor) for backend in ("cudf",)):
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(nw_eager_constructor):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")
    df = nw.from_native(nw_eager_constructor(data), eager_only=True)
    result = df["a"].cast(nw.List(nw.Int32())).list.min().to_list()
    assert_equal_data({"a": result}, {"a": expected})
