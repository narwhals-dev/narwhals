from __future__ import annotations

import numpy as np
import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from tests.utils import Constructor
from tests.utils import assert_equal_data

arr = np.array([[5, 2, 0, 1], [1, 4, 7, 8], [1, 2, 3, 9]])


def test_from_numpy(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
    native_namespace = nw.get_native_namespace(df)
    result = nw.from_numpy(arr, native_namespace=native_namespace)
    expected = {
        "column_0": [5, 2, 0, 1],
        "column_1": [1, 4, 7, 8],
        "column_2": [1, 2, 3, 9],
    }
    assert_equal_data(result, expected)
    assert isinstance(result, nw.DataFrame)


def test_from_numpy_schema(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    schema = {"c": nw_v1.Int16(), "d": nw_v1.Float32(), "e": nw_v1.Int16()}
    df = nw_v1.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
    native_namespace = nw_v1.get_native_namespace(df)
    result = nw_v1.from_numpy(
        arr,
        native_namespace=native_namespace,
        schema=schema,  # type: ignore[arg-type]
    )
    assert result.collect_schema() == schema


def test_from_dict_without_namespace_invalid() -> None:
    with pytest.raises(TypeError, match="namespace"):
        nw.from_numpy(arr)


def test_from_numpy_v1(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw_v1.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
    native_namespace = nw_v1.get_native_namespace(df)
    result = nw_v1.from_numpy(arr, native_namespace=native_namespace)
    expected = {
        "column_0": [5, 2, 0, 1],
        "column_1": [1, 4, 7, 8],
        "column_2": [1, 2, 3, 9],
    }
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v1.DataFrame)


def test_from_numpy_empty() -> None:
    with pytest.raises(ValueError, match="`from_numpy` only accepts 2D numpy arrays"):
        nw.from_numpy(np.array([0]))
