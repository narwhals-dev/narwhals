from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, POLARS_VERSION, assert_equal_data, is_windows

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager

data = {"a": [[3, None, 2, 2, 4, None], [-1], None, [None, None, None], [], [3, 4, None]]}
expected = [2.5, -1, None, None, None, 3.5]
expected_pyarrow = [2.5, -1, None, None, None, 3]


def test_median_expr(
    request: pytest.FixtureRequest, nw_frame_constructor: Constructor
) -> None:
    if any(
        backend in str(nw_frame_constructor) for backend in ("dask", "cudf", "ibis")
    ) or ("polars" in str(nw_frame_constructor) and POLARS_VERSION < (0, 20, 7)):
        # ibis issue: https://github.com/ibis-project/ibis/issues/11788
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(nw_frame_constructor):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")
    if (
        any(backend in str(nw_frame_constructor) for backend in ("pandas", "pyarrow"))
        and sys.version_info < (3, 10)
        and is_windows()
    ):  # pragma: no cover
        reason = "The issue only affects old Python versions on Windows."
        request.applymarker(pytest.mark.xfail(reason=reason))
    result = nw.from_native(nw_frame_constructor(data)).select(
        nw.col("a").cast(nw.List(nw.Int32())).list.median()
    )
    if any(
        backend in str(nw_frame_constructor)
        for backend in ("pandas", "pyarrow", "pandas[pyarrow]")
    ):
        # there is a mismatch as pyarrow uses an approximate median
        assert_equal_data(result, {"a": expected_pyarrow})
    else:
        assert_equal_data(result, {"a": expected})


def test_median_series(
    request: pytest.FixtureRequest, nw_eager_constructor: ConstructorEager
) -> None:
    if any(backend in str(nw_eager_constructor) for backend in ("cudf",)) or (
        "polars" in str(nw_eager_constructor) and POLARS_VERSION < (0, 20, 7)
    ):
        request.applymarker(pytest.mark.xfail)
    if "pandas" in str(nw_eager_constructor):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")
    if (
        any(backend in str(nw_eager_constructor) for backend in ("pandas", "pyarrow"))
        and sys.version_info < (3, 10)
        and is_windows()
    ):  # pragma: no cover
        reason = "The issue only affects old Python versions on Windows."
        request.applymarker(pytest.mark.xfail(reason=reason))
    df = nw.from_native(nw_eager_constructor(data), eager_only=True)
    result = df["a"].cast(nw.List(nw.Int32())).list.median()
    if any(
        backend in str(nw_eager_constructor)
        for backend in ("pandas", "pyarrow", "pandas[pyarrow]")
    ):
        # there is a mismatch as pyarrow uses an approximate median
        assert_equal_data({"a": result}, {"a": expected_pyarrow})
    else:
        assert_equal_data({"a": result}, {"a": expected})
