from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, POLARS_VERSION, assert_equal_data, is_windows

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager

data = {"a": [[3, None, 2, 2, 4, None], [-1], None, [None, None, None], [], [3, 4, None]]}


@pytest.mark.parametrize(
    ("index", "expected"), [(0, 2.5), (1, -1), (2, None), (3, None), (4, None), (5, 3.5)]
)
def test_median_expr(
    request: pytest.FixtureRequest, constructor: Constructor, index: int, expected: float
) -> None:
    if any(
        backend in str(constructor) for backend in ("dask", "cudf", "sqlframe", "ibis")
    ) or ("polars" in str(constructor) and POLARS_VERSION < (0, 20, 7)):
        # sqlframe issue: https://github.com/eakmanrq/sqlframe/issues/548
        # ibis issue: https://github.com/ibis-project/ibis/issues/11788
        request.applymarker(pytest.mark.xfail)
    if os.environ.get("SPARK_CONNECT", None) and "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(constructor):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")
    if (
        any(backend in str(constructor) for backend in ("pandas", "pyarrow"))
        and sys.version_info < (3, 10)
        and is_windows
    ):  # pragma: no cover
        reason = "The issue only affects old Python versions on Windows."
        pytest.skip(reason=reason)
    result = (
        nw.from_native(constructor(data))
        .select(nw.col("a").cast(nw.List(nw.Int32())).list.median())
        .lazy()
        .collect()["a"]
        .to_list()
    )
    if any(
        backend in str(constructor)
        for backend in ("pandas", "pyarrow", "pandas[pyarrow]")
    ) and (index == 5):
        # there is a mismatch as pyarrow uses an approximate median
        assert_equal_data({"a": [result[index]]}, {"a": [3]})
    else:
        assert_equal_data({"a": [result[index]]}, {"a": [expected]})


@pytest.mark.parametrize(
    ("index", "expected"), [(0, 2.5), (1, -1), (2, None), (3, None), (4, None), (5, 3.5)]
)
def test_median_series(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    index: int,
    expected: float,
) -> None:
    if any(backend in str(constructor_eager) for backend in ("cudf",)) or (
        "polars" in str(constructor_eager) and POLARS_VERSION < (0, 20, 7)
    ):
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(constructor_eager):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")
    if (
        any(backend in str(constructor_eager) for backend in ("pandas", "pyarrow"))
        and sys.version_info < (3, 10)
        and is_windows
    ):  # pragma: no cover
        reason = "The issue only affects old Python versions on Windows."
        pytest.skip(reason=reason)
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df["a"].cast(nw.List(nw.Int32())).list.median().to_list()
    if any(
        backend in str(constructor_eager)
        for backend in ("pandas", "pyarrow", "pandas[pyarrow]")
    ) and (index == 5):
        # there is a mismatch as pyarrow uses an approximate median
        assert_equal_data({"a": [result[index]]}, {"a": [3]})
    else:
        assert_equal_data({"a": [result[index]]}, {"a": [expected]})
