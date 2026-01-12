from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, assert_equal_data

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager

data = {"a": [[3, None, 2, 2, 4, None], [-1], None, [None, None, None], []]}
expected = [2.75, -1, None, None, None]


def test_mean_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(backend in str(constructor) for backend in ("dask", "cudf")):
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(constructor):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")
    result = nw.from_native(constructor(data)).select(
        nw.col("a").cast(nw.List(nw.Int32())).list.mean()
    )
    assert_equal_data(result, {"a": expected})


def test_mean_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if any(backend in str(constructor_eager) for backend in ("cudf",)):
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(constructor_eager):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df["a"].cast(nw.List(nw.Int32())).list.mean()
    assert_equal_data({"a": result}, {"a": expected})
