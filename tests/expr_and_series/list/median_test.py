from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, POLARS_VERSION

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager

data = {"a": [[3, 2, 2, 4, None], [-1]]}


def test_median_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(
        backend in str(constructor)
        for backend in ("dask", "cudf", "sqlframe", "ibis", "pyspark")
    ) or ("polars" in str(constructor) and POLARS_VERSION < (0, 20, 7)):
        # PySpark issue: https://issues.apache.org/jira/browse/SPARK-54382
        # ibis issue: https://github.com/ibis-project/ibis/issues/11788
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(constructor):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")
    result = (
        nw.from_native(constructor(data))
        .select(nw.col("a").cast(nw.List(nw.Int32())).list.median())
        .lazy()
        .collect()["a"]
        .to_list()
    )
    assert result[0] == 2.5
    assert result[1] == -1


def test_median_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if any(backend in str(constructor_eager) for backend in ("cudf",)) or (
        "polars" in str(constructor_eager) and POLARS_VERSION < (0, 20, 7)
    ):
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(constructor_eager):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df["a"].cast(nw.List(nw.Int32())).list.median().to_list()
    assert result[0] == 2.5
    assert result[1] == -1
