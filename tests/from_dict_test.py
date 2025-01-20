from __future__ import annotations

from datetime import datetime

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from tests.utils import Constructor
from tests.utils import assert_equal_data


def test_from_dict(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "dask" in str(constructor) or "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
    native_namespace = nw.get_native_namespace(df)
    result = nw.from_dict({"c": [1, 2], "d": [5, 6]}, native_namespace=native_namespace)
    expected = {"c": [1, 2], "d": [5, 6]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw.DataFrame)


def test_from_dict_schema(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "dask" in str(constructor) or "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    schema = {"c": nw_v1.Int16(), "d": nw_v1.Float32()}
    df = nw_v1.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
    native_namespace = nw_v1.get_native_namespace(df)
    result = nw_v1.from_dict(
        {"c": [1, 2], "d": [5, 6]},
        native_namespace=native_namespace,
        schema=schema,  # type: ignore[arg-type]
    )
    assert result.collect_schema() == schema


def test_from_dict_without_namespace(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]})).lazy().collect()
    result = nw.from_dict({"c": df["a"], "d": df["b"]})
    assert_equal_data(result, {"c": [1, 2, 3], "d": [4, 5, 6]})


def test_from_dict_without_namespace_invalid(
    constructor: Constructor,
) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]})).lazy().collect()
    with pytest.raises(TypeError, match="namespace"):
        nw.from_dict({"c": nw.to_native(df["a"]), "d": nw.to_native(df["b"])})


def test_from_dict_one_native_one_narwhals(
    constructor: Constructor,
) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]})).lazy().collect()
    result = nw.from_dict({"c": nw.to_native(df["a"]), "d": df["b"]})
    expected = {"c": [1, 2, 3], "d": [4, 5, 6]}
    assert_equal_data(result, expected)


def test_from_dict_v1(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "dask" in str(constructor) or "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw_v1.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
    native_namespace = nw_v1.get_native_namespace(df)
    result = nw_v1.from_dict(
        {"c": [1, 2], "d": [datetime(2020, 1, 1), datetime(2020, 1, 2)]},
        native_namespace=native_namespace,
    )
    expected = {"c": [1, 2], "d": [datetime(2020, 1, 1), datetime(2020, 1, 2)]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v1.DataFrame)
    assert isinstance(result.schema["d"], nw_v1.dtypes.Datetime)


def test_from_dict_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        nw.from_dict({})


def test_alignment() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    # https://github.com/narwhals-dev/narwhals/issues/1474
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = nw.from_dict(
        {"a": df["a"], "b": df["a"].sort_values(ascending=False)}, native_namespace=pd
    ).to_native()
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    pd.testing.assert_frame_equal(result, expected)
