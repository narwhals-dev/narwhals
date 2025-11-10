from __future__ import annotations

import re
from typing import Any

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, Constructor, ConstructorEager, assert_equal_data

data = {"a": [[1, 2], [None, 3], [None], None]}


@pytest.mark.parametrize(("index", "expected"), [(0, {"a": [1, None, None, None]})])
def test_get_expr(
    request: pytest.FixtureRequest, constructor: Constructor, index: int, expected: Any
) -> None:
    if any(backend in str(constructor) for backend in ("dask", "cudf")):
        request.applymarker(pytest.mark.xfail)

    if "pandas" in str(constructor):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")

    result = nw.from_native(constructor(data)).select(
        nw.col("a").cast(nw.List(nw.Int32())).list.get(index)
    )

    assert_equal_data(result, expected)


@pytest.mark.parametrize(("index", "expected"), [(0, {"a": [1, None, None, None]})])
def test_get_series(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    index: int,
    expected: Any,
) -> None:
    if "cudf" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    if "pandas" in str(constructor_eager):
        if PANDAS_VERSION < (2, 2):
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


def test_get_expr_negative_index(constructor: Constructor) -> None:
    data = {"a": [[1, 2], [None, 3], [None], None]}
    index = -1

    if "pandas" in str(constructor):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")

    df = nw.from_native(constructor(data))
    msg = re.escape(
        f"Index {index} is out of bounds: should be greater than or equal to 0."
    )
    with pytest.raises(ValueError, match=msg):
        df.select(nw.col("a").cast(nw.List(nw.Int32())).list.get(index))


def test_get_series_negative_index(constructor_eager: ConstructorEager) -> None:
    data = {"a": [[1, 2], [None, 3], [None], None]}
    index = -1

    if "pandas" in str(constructor_eager):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")

    df = nw.from_native(constructor_eager(data), eager_only=True)
    msg = re.escape(
        f"Index {index} is out of bounds: should be greater than or equal to 0."
    )
    with pytest.raises(ValueError, match=msg):
        df["a"].list.get(index)


def test_get_expr_non_int_index(constructor: Constructor) -> None:
    data = {"a": [[1, 2], [None, 3], [None], None], "index": [0, 1, 0, 0]}
    index = "index"

    if "pandas" in str(constructor):
        if PANDAS_VERSION < (2, 2):
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
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")

    df = nw.from_native(constructor_eager(data), eager_only=True)
    msg = re.escape(
        f"Index must be of type 'int'. Got type '{type(index).__name__}' instead."
    )
    with pytest.raises(TypeError, match=msg):
        df["a"].list.get(index)  # type: ignore[arg-type]
