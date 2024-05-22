from typing import Any

import pandas as pd
import polars as pl
import pytest

import narwhals as nw
from tests.utils import compare_dicts

data = {
    "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "b": [11, 12, 13, 14, 15, 16],
}


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_slice_column(constructor: Any) -> None:
    result = nw.from_native(constructor(data))["a"]
    assert isinstance(result, nw.Series)
    assert result.to_numpy().tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_slice_rows(constructor: Any) -> None:
    result = nw.from_native(constructor(data))[1:]
    compare_dicts(result, {"a": [2.0, 3.0, 4.0, 5.0, 6.0], "b": [12, 13, 14, 15, 16]})

    result = nw.from_native(constructor(data))[2:4]
    compare_dicts(result, {"a": [3.0, 4.0], "b": [13, 14]})


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_slice_rows_with_step(constructor: Any) -> None:
    result = nw.from_native(constructor(data))[1::2]
    compare_dicts(result, {"a": [2.0, 4.0, 6.0], "b": [12, 14, 16]})


@pytest.mark.parametrize("constructor", [pl.LazyFrame])
def test_slice_lazy_fails(constructor: Any) -> None:
    with pytest.raises(TypeError, match="Slicing is not supported on LazyFrame"):
        _ = nw.from_native(constructor(data))[1:]


@pytest.mark.parametrize("constructor", [pd.DataFrame, pl.DataFrame])
def test_slice_int_fails(constructor: Any) -> None:
    with pytest.raises(
        TypeError, match="Expected str, range or slice, got: <class 'int'>"
    ):
        _ = nw.from_native(constructor(data))[1]  # type: ignore[call-overload,index]
