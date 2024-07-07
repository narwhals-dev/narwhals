from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {
    "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "b": [11, 12, 13, 14, 15, 16],
}


def test_slice_column(constructor_with_pyarrow: Any) -> None:
    result = nw.from_native(constructor_with_pyarrow(data))["a"]
    assert isinstance(result, nw.Series)
    assert result.to_numpy().tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_slice_rows(constructor_with_pyarrow: Any) -> None:
    result = nw.from_native(constructor_with_pyarrow(data))[1:]
    compare_dicts(result, {"a": [2.0, 3.0, 4.0, 5.0, 6.0], "b": [12, 13, 14, 15, 16]})

    result = nw.from_native(constructor_with_pyarrow(data))[2:4]
    compare_dicts(result, {"a": [3.0, 4.0], "b": [13, 14]})


def test_slice_rows_with_step(constructor: Any) -> None:
    result = nw.from_native(constructor(data))[1::2]
    compare_dicts(result, {"a": [2.0, 4.0, 6.0], "b": [12, 14, 16]})


def test_slice_rows_with_step_pyarrow() -> None:
    with pytest.raises(
        NotImplementedError, match="Slicing with step is not supported on PyArrow tables"
    ):
        nw.from_native(pa.table(data))[1::2]


def test_slice_lazy_fails() -> None:
    with pytest.raises(TypeError, match="Slicing is not supported on LazyFrame"):
        _ = nw.from_native(pl.LazyFrame(data))[1:]


def test_slice_int_fails(constructor_with_pyarrow: Any) -> None:
    with pytest.raises(TypeError, match="Expected str or slice, got: <class 'int'>"):
        _ = nw.from_native(constructor_with_pyarrow(data))[1]  # type: ignore[call-overload,index]


def test_gather(constructor_with_pyarrow: Any) -> None:
    df = nw.from_native(constructor_with_pyarrow(data), eager_only=True)
    result = df[[0, 3, 1]]
    expected = {
        "a": [1.0, 4.0, 2.0],
        "b": [11, 14, 12],
    }
    compare_dicts(result, expected)
    result = df[np.array([0, 3, 1])]
    compare_dicts(result, expected)


def test_gather_pandas_index() -> None:
    # check that we're slicing positionally, and not on the pandas index
    df = pd.DataFrame({"a": [4, 1, 2], "b": [1, 4, 2]}, index=[2, 1, 3])
    result = nw.from_native(df, eager_only=True)[[1, 2]]
    expected = {"a": [1, 2], "b": [4, 2]}
    compare_dicts(result, expected)
