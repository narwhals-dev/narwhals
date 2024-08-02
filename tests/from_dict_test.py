from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_from_dict(constructor: Any, request: Any) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
    native_namespace = nw.get_native_namespace(df)
    result = nw.from_dict({"c": [1, 2], "d": [5, 6]}, native_namespace=native_namespace)
    expected = {"c": [1, 2], "d": [5, 6]}
    compare_dicts(result, expected)
    assert isinstance(result, nw.DataFrame)


def test_from_dict_schema(constructor: Any, request: Any) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    schema = {"c": nw.Int16(), "d": nw.Float32()}
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
    native_namespace = nw.get_native_namespace(df)
    result = nw.from_dict(
        {"c": [1, 2], "d": [5, 6]},
        native_namespace=native_namespace,
        schema=schema,  # type: ignore[arg-type]
    )
    assert result.collect_schema() == schema
