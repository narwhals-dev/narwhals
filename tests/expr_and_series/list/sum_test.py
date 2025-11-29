from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager

data = {"a": [[3, None, 2, 2, 4, None], [], [-1], [None, None, None], []]}


@pytest.mark.parametrize(
    ("index", "expected"), [(0, 11), (1, 0), (2, -1), (3, 0), (4, 0)]
)
def test_sum_expr(
    request: pytest.FixtureRequest, constructor: Constructor, index: int, expected: int
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
        .select(nw.col("a").cast(nw.List(nw.Int32())).list.sum())
        .lazy()
        .collect()["a"]
        .to_list()
    )
    assert result[index] == expected


@pytest.mark.parametrize(
    ("index", "expected"), [(0, 11), (1, 0), (2, -1), (3, 0), (4, 0)]
)
def test_sum_series(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    index: int,
    expected: int,
) -> None:
    if any(backend in str(constructor_eager) for backend in ("cudf",)):
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(constructor_eager):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df["a"].cast(nw.List(nw.Int32())).list.sum().to_list()
    assert result[index] == expected
