from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, assert_equal_data

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager

data = {"a": [[3, None, 2, 2, 4, None], [-1], None, [None, None, None], []]}


@pytest.mark.parametrize(
    ("index", "expected"), [(0, 2.75), (1, -1), (2, None), (3, None), (4, None)]
)
def test_mean_expr(
    request: pytest.FixtureRequest, constructor: Constructor, index: int, expected: float
) -> None:
    if any(backend in str(constructor) for backend in ("dask", "cudf", "sqlframe")):
        # sqlframe issue: https://github.com/eakmanrq/sqlframe/issues/548
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(constructor):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")
    result = (
        nw.from_native(constructor(data))
        .select(nw.col("a").cast(nw.List(nw.Int32())).list.mean())
        .lazy()
        .collect()["a"]
        .to_list()
    )
    assert_equal_data({"a": [result[index]]}, {"a": [expected]})


@pytest.mark.parametrize(
    ("index", "expected"), [(0, 2.75), (1, -1), (2, None), (3, None), (4, None)]
)
def test_mean_series(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    index: int,
    expected: float,
) -> None:
    if any(backend in str(constructor_eager) for backend in ("cudf",)):
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(constructor_eager):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df["a"].cast(nw.List(nw.Int32())).list.mean().to_list()
    assert_equal_data({"a": [result[index]]}, {"a": [expected]})
