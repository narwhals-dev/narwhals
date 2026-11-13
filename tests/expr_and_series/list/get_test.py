from __future__ import annotations

import re
from typing import Any

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from narwhals.exceptions import ComputeError
from tests.utils import (
    PANDAS_VERSION,
    POLARS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

data = {"a": [[1, 2], [None, 3], [None], None, [4]]}


@pytest.mark.parametrize(
    ("index", "expected"),
    [(0, {"a": [1, None, None, None, 4]}), (-1, {"a": [2, 3, None, None, 4]})],
)
def test_get_expr(
    request: pytest.FixtureRequest, constructor: Constructor, index: int, expected: Any
) -> None:
    if "cudf" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    if any(backend in str(constructor) for backend in ("pandas", "dask")):
        if PANDAS_VERSION < PkgVersion("2.2"):
            pytest.skip()
        pytest.importorskip("pyarrow")

    result = nw.from_native(constructor(data)).select(
        nw.col("a").cast(nw.List(nw.Int32())).list.get(index)
    )

    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("index", "expected"),
    [(0, {"a": [1, None, None, None, 4]}), (-1, {"a": [2, 3, None, None, 4]})],
)
def test_get_series(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    index: int,
    expected: Any,
) -> None:
    if "cudf" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    if "pandas" in str(constructor_eager):
        if PANDAS_VERSION < PkgVersion("2.2"):
            pytest.skip()
        pytest.importorskip("pyarrow")

    if (
        constructor_eager.__name__.startswith("pandas")
        and "pyarrow" not in constructor_eager.__name__
    ):
        df = nw.from_native(constructor_eager(data), eager_only=True)
        msg = re.escape("Series must be of PyArrow List type to support list namespace.")
        with pytest.raises(TypeError, match=msg):
            df["a"].list.get(index)
        return

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df["a"].cast(nw.List(nw.Int32())).list.get(index)

    assert_equal_data({"a": result}, expected)


@pytest.mark.parametrize("index", [-2, 1])
def test_get_out_of_bounds(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager, index: int
) -> None:
    if "cudf" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
    if "polars" in str(constructor_eager) and POLARS_VERSION < PkgVersion("1"):
        pytest.skip(reason="older Polars returns null out of bounds")
    if any(x in str(constructor_eager) for x in ("pandas", "modin")):
        if index >= 0:
            pytest.skip(reason="pandas raises its own error for a non-negative index")
        if PANDAS_VERSION < PkgVersion("2.2"):
            pytest.skip()
        pytest.importorskip("pyarrow")

    df = nw.from_native(constructor_eager({"a": [[1, 2], [3]]}), eager_only=True)
    with pytest.raises(ComputeError, match="out of bounds"):
        df.select(nw.col("a").cast(nw.List(nw.Int32())).list.get(index))


def test_get_negative_index_fixed_size_list() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    native = pa.chunked_array([pa.array([[1, 2], None, [5, 6]], pa.list_(pa.int64(), 2))])
    s = nw.from_native(native, series_only=True)
    assert_equal_data({"a": s.list.get(-1)}, {"a": [2, None, 6]})
    with pytest.raises(ComputeError, match="out of bounds"):
        s.list.get(-3)


def test_get_negative_index_null_list_spanning_values() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    # The null list's offsets still span the values `3, 4`.
    native = pa.ListArray.from_arrays(
        pa.array([0, 2, 4, 5], pa.int32()),
        pa.array([1, 2, 3, 4, 5]),
        mask=pa.array([False, True, False]),
    )
    s = nw.from_native(pa.chunked_array([native]), series_only=True)
    assert_equal_data({"a": s.list.get(-1)}, {"a": [2, None, 5]})


def test_get_expr_non_int_index(constructor: Constructor) -> None:
    data = {"a": [[1, 2], [None, 3], [None], None], "index": [0, 1, 0, 0]}
    index = "index"

    if "pandas" in str(constructor):
        if PANDAS_VERSION < PkgVersion("2.2"):
            pytest.skip()
        pytest.importorskip("pyarrow")

    df = nw.from_native(constructor(data))
    msg = re.escape(
        f"Index must be of type 'int'. Got type '{type(index).__name__}' instead."
    )
    with pytest.raises(TypeError, match=msg):
        df.select(nw.col("a").cast(nw.List(nw.Int32())).list.get(index))  # type: ignore[arg-type]


def test_get_series_non_int_index(constructor_eager: ConstructorEager) -> None:
    data = {"a": [[1, 2], [None, 3], [None], None], "index": [0, 1, 0, 0]}
    index = "index"

    if "pandas" in str(constructor_eager):
        if PANDAS_VERSION < PkgVersion("2.2"):
            pytest.skip()
        pytest.importorskip("pyarrow")

    df = nw.from_native(constructor_eager(data), eager_only=True)
    msg = re.escape(
        f"Index must be of type 'int'. Got type '{type(index).__name__}' instead."
    )
    with pytest.raises(TypeError, match=msg):
        df["a"].list.get(index)  # type: ignore[arg-type]
