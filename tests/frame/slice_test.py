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


def test_slice_column(constructor_eager: Any) -> None:
    result = nw.from_native(constructor_eager(data))["a"]
    assert isinstance(result, nw.Series)
    compare_dicts({"a": result}, {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})


def test_slice_rows(constructor_eager: Any) -> None:
    result = nw.from_native(constructor_eager(data))[1:]
    compare_dicts(result, {"a": [2.0, 3.0, 4.0, 5.0, 6.0], "b": [12, 13, 14, 15, 16]})

    result = nw.from_native(constructor_eager(data))[2:4]
    compare_dicts(result, {"a": [3.0, 4.0], "b": [13, 14]})


def test_slice_rows_with_step(request: Any, constructor_eager: Any) -> None:
    if "pyarrow_table" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
    result = nw.from_native(constructor_eager(data))[1::2]
    compare_dicts(result, {"a": [2.0, 4.0, 6.0], "b": [12, 14, 16]})


def test_slice_rows_with_step_pyarrow() -> None:
    with pytest.raises(
        NotImplementedError,
        match="Slicing with step is not supported on PyArrow tables",
    ):
        nw.from_native(pa.table(data))[1::2]


def test_slice_lazy_fails() -> None:
    with pytest.raises(TypeError, match="Slicing is not supported on LazyFrame"):
        _ = nw.from_native(pl.LazyFrame(data))[1:]


def test_slice_int_fails(constructor_eager: Any) -> None:
    with pytest.raises(TypeError, match="Expected str or slice, got: <class 'int'>"):
        _ = nw.from_native(constructor_eager(data))[1]  # type: ignore[call-overload,index]


def test_gather(constructor_eager: Any) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
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

    result = nw.from_native(df, eager_only=True)[[1, 2], "a"].to_frame()
    expected = {"a": [1, 2]}
    compare_dicts(result, expected)


def test_gather_rows_cols(constructor_eager: Any) -> None:
    native_df = constructor_eager(data)
    df = nw.from_native(native_df, eager_only=True)

    expected = {"b": [11, 14, 12]}

    result = {"b": df[[0, 3, 1], 1]}
    compare_dicts(result, expected)

    result = {"b": df[np.array([0, 3, 1]), "b"]}
    compare_dicts(result, expected)


def test_slice_both_tuples_of_ints(constructor_eager: Any) -> None:
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df[[0, 1], [0, 2]]
    expected = {"a": [1, 2], "c": [7, 8]}
    compare_dicts(result, expected)


def test_slice_int_rows_str_columns(constructor_eager: Any) -> None:
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df[[0, 1], ["a", "c"]]
    expected = {"a": [1, 2], "c": [7, 8]}
    compare_dicts(result, expected)


def test_slice_invalid(constructor_eager: Any) -> None:
    data = {"a": [1, 2], "b": [4, 5]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    with pytest.raises(TypeError, match="Hint:"):
        df[0, 0]
