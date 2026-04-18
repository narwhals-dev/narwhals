from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager

data = {"a": [[2, 2, 3, None, None], None, []]}
expected = {"a": [True, None, False]}


def test_contains_expr(
    request: pytest.FixtureRequest, nw_frame_constructor: Constructor
) -> None:
    if any(
        backend in str(nw_frame_constructor)
        for backend in ("dask", "modin", "cudf", "pyarrow", "pandas")
    ):
        request.applymarker(pytest.mark.xfail)
    result = nw.from_native(nw_frame_constructor(data)).select(
        nw.col("a").cast(nw.List(nw.Int32())).list.contains(2)
    )
    assert_equal_data(result, expected)


def test_contains_series(
    request: pytest.FixtureRequest, nw_eager_constructor: ConstructorEager
) -> None:
    if any(
        backend in str(nw_eager_constructor)
        for backend in ("modin", "cudf", "pyarrow", "pandas")
    ):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(nw_eager_constructor(data), eager_only=True)
    result = df["a"].cast(nw.List(nw.Int32())).list.contains(2)
    assert_equal_data({"a": result}, expected)
