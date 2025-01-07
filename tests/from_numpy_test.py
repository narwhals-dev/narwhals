from __future__ import annotations

import numpy as np
import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {"a": [1, 2, 3], "b": [4, 5, 6]}
arr = np.array([[5, 2, 0, 1], [1, 4, 7, 8], [1, 2, 3, 9]])
expected = {
    "column_0": [5, 1, 1],
    "column_1": [2, 4, 2],
    "column_2": [0, 7, 3],
    "column_3": [1, 8, 9],
}


def test_from_numpy(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    native_namespace = nw.get_native_namespace(df)
    result = nw.from_numpy(arr, native_namespace=native_namespace)
    assert_equal_data(result, expected)
    assert isinstance(result, nw.DataFrame)


def test_from_numpy_schema_dict(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    schema = {
        "c": nw_v1.Int16(),
        "d": nw_v1.Float32(),
        "e": nw_v1.Int16(),
        "f": nw_v1.Float64(),
    }
    df = nw_v1.from_native(constructor(data))
    native_namespace = nw_v1.get_native_namespace(df)
    result = nw_v1.from_numpy(
        arr,
        native_namespace=native_namespace,
        schema=schema,  # type: ignore[arg-type]
    )
    assert result.collect_schema() == schema


def test_from_numpy_schema_list(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    schema = ["c", "d", "e", "f"]
    df = nw_v1.from_native(constructor(data))
    native_namespace = nw_v1.get_native_namespace(df)
    result = nw_v1.from_numpy(arr, native_namespace=native_namespace, schema=schema)
    assert result.columns == schema


def test_from_numpy_schema_notvalid(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    native_namespace = nw_v1.get_native_namespace(df)
    with pytest.raises(
        TypeError, match="`schema` is expected to be one of the following types"
    ):
        nw.from_numpy(arr, schema="a", native_namespace=native_namespace)  # type: ignore[arg-type]


def test_from_numpy_v1(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw_v1.from_native(constructor(data))
    native_namespace = nw_v1.get_native_namespace(df)
    result = nw_v1.from_numpy(arr, native_namespace=native_namespace)
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v1.DataFrame)


def test_from_numpy_not2d(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    native_namespace = nw_v1.get_native_namespace(df)
    with pytest.raises(ValueError, match="`from_numpy` only accepts 2D numpy arrays"):
        nw.from_numpy(np.array([0]), native_namespace=native_namespace)
