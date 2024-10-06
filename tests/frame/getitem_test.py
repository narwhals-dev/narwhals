from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from tests.utils import ConstructorEager
from tests.utils import compare_dicts

data = {
    "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "b": [11, 12, 13, 14, 15, 16],
}


def test_slice_column(constructor_eager: ConstructorEager) -> None:
    result = nw.from_native(constructor_eager(data))["a"]
    assert isinstance(result, nw.Series)
    compare_dicts({"a": result}, {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})


def test_slice_rows(constructor_eager: ConstructorEager) -> None:
    result = nw.from_native(constructor_eager(data))[1:]
    compare_dicts(result, {"a": [2.0, 3.0, 4.0, 5.0, 6.0], "b": [12, 13, 14, 15, 16]})

    result = nw.from_native(constructor_eager(data))[2:4]
    compare_dicts(result, {"a": [3.0, 4.0], "b": [13, 14]})


def test_slice_rows_with_step(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
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


def test_slice_int(constructor_eager: ConstructorEager) -> None:
    result = nw.from_native(constructor_eager(data), eager_only=True)[1]  # type: ignore[call-overload]
    compare_dicts(result, {"a": [2], "b": [12]})


def test_slice_fails(constructor_eager: ConstructorEager) -> None:
    class Foo: ...

    with pytest.raises(TypeError, match="Expected str or slice, got:"):
        nw.from_native(constructor_eager(data), eager_only=True)[Foo()]  # type: ignore[call-overload]


def test_gather(constructor_eager: ConstructorEager) -> None:
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


def test_gather_rows_cols(constructor_eager: ConstructorEager) -> None:
    native_df = constructor_eager(data)
    df = nw.from_native(native_df, eager_only=True)

    expected = {"b": [11, 14, 12]}

    result = {"b": df[[0, 3, 1], 1]}
    compare_dicts(result, expected)

    result = {"b": df[np.array([0, 3, 1]), "b"]}
    compare_dicts(result, expected)


def test_slice_both_tuples_of_ints(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df[[0, 1], [0, 2]]
    expected = {"a": [1, 2], "c": [7, 8]}
    compare_dicts(result, expected)


def test_slice_int_rows_str_columns(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df[[0, 1], ["a", "c"]]
    expected = {"a": [1, 2], "c": [7, 8]}
    compare_dicts(result, expected)


def test_slice_slice_columns(constructor_eager: ConstructorEager) -> None:  # noqa: PLR0915
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [1, 4, 2]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df[[0, 1], "b":"c"]  # type: ignore[misc]
    expected = {"b": [4, 5], "c": [7, 8]}
    compare_dicts(result, expected)
    result = df[[0, 1], :"c"]  # type: ignore[misc]
    expected = {"a": [1, 2], "b": [4, 5], "c": [7, 8]}
    compare_dicts(result, expected)
    result = df[[0, 1], "a":"d":2]  # type: ignore[misc]
    expected = {"a": [1, 2], "c": [7, 8]}
    compare_dicts(result, expected)
    result = df[[0, 1], "b":]  # type: ignore[misc]
    expected = {"b": [4, 5], "c": [7, 8], "d": [1, 4]}
    compare_dicts(result, expected)
    result = df[[0, 1], 1:3]
    expected = {"b": [4, 5], "c": [7, 8]}
    compare_dicts(result, expected)
    result = df[[0, 1], :3]
    expected = {"a": [1, 2], "b": [4, 5], "c": [7, 8]}
    compare_dicts(result, expected)
    result = df[[0, 1], 0:4:2]
    expected = {"a": [1, 2], "c": [7, 8]}
    compare_dicts(result, expected)
    result = df[[0, 1], 1:]
    expected = {"b": [4, 5], "c": [7, 8], "d": [1, 4]}
    compare_dicts(result, expected)
    result = df[:, ["b", "d"]]
    expected = {"b": [4, 5, 6], "d": [1, 4, 2]}
    compare_dicts(result, expected)
    result = df[:, [0, 2]]
    expected = {"a": [1, 2, 3], "c": [7, 8, 9]}
    compare_dicts(result, expected)
    result = df[:2, [0, 2]]
    expected = {"a": [1, 2], "c": [7, 8]}
    compare_dicts(result, expected)
    result = df[:2, ["a", "c"]]
    expected = {"a": [1, 2], "c": [7, 8]}
    compare_dicts(result, expected)
    result = df[1:, [0, 2]]
    expected = {"a": [2, 3], "c": [8, 9]}
    compare_dicts(result, expected)
    result = df[1:, ["a", "c"]]
    expected = {"a": [2, 3], "c": [8, 9]}
    compare_dicts(result, expected)
    result = df[["b", "c"]]
    expected = {"b": [4, 5, 6], "c": [7, 8, 9]}
    compare_dicts(result, expected)
    result = df[:2]
    expected = {"a": [1, 2], "b": [4, 5], "c": [7, 8], "d": [1, 4]}
    compare_dicts(result, expected)
    result = df[2:]
    expected = {"a": [3], "b": [6], "c": [9], "d": [2]}
    compare_dicts(result, expected)
    # mypy says "Slice index must be an integer", but we do in fact support
    # using string slices
    result = df["a":"b"]  # type: ignore[misc]
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    compare_dicts(result, expected)
    result = df[(0, 1), :]
    expected = {"a": [1, 2], "b": [4, 5], "c": [7, 8], "d": [1, 4]}
    compare_dicts(result, expected)
    result = df[[0, 1], :]
    expected = {"a": [1, 2], "b": [4, 5], "c": [7, 8], "d": [1, 4]}
    compare_dicts(result, expected)
    result = df[[0, 1], df.columns]
    expected = {"a": [1, 2], "b": [4, 5], "c": [7, 8], "d": [1, 4]}
    compare_dicts(result, expected)


def test_slice_invalid(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 2], "b": [4, 5]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    with pytest.raises(TypeError, match="Hint:"):
        df[0, 0]


def test_slice_edge_cases(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [1, 4, 2]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    assert df[[], :].shape == (0, 4)
    assert df[:, []].shape == (0, 0)
    assert df[[]].shape == (0, 4)
    assert df[[], ["a"]].shape == (0, 1)
    assert df[:, :].shape == (3, 4)
    assert df[[], []].shape == (0, 0)


@pytest.mark.parametrize(
    ("row_idx", "col_idx"),
    [
        ([0, 2], [0]),
        ((0, 2), [0]),
        ([0, 2], (0,)),
        ((0, 2), (0,)),
        ([0, 2], range(1)),
        (range(2), [0]),
        (range(2), range(1)),
    ],
)
def test_get_item_works_with_tuple_and_list_and_range_row_and_col_indexing(
    constructor_eager: ConstructorEager,
    row_idx: list[int] | tuple[int] | range,
    col_idx: list[int] | tuple[int] | range,
) -> None:
    nw_df = nw.from_native(constructor_eager(data), eager_only=True)
    nw_df[row_idx, col_idx]


@pytest.mark.parametrize(
    ("row_idx", "col"),
    [
        ([0, 2], slice(1)),
        ((0, 2), slice(1)),
        (range(2), slice(1)),
    ],
)
def test_get_item_works_with_tuple_and_list_and_range_row_indexing_and_slice_col_indexing(
    constructor_eager: ConstructorEager,
    row_idx: list[int] | tuple[int] | range,
    col: slice,
) -> None:
    nw_df = nw.from_native(constructor_eager(data), eager_only=True)
    nw_df[row_idx, col]


@pytest.mark.parametrize(
    ("row_idx", "col"),
    [
        ([0, 2], "a"),
        ((0, 2), "a"),
        (range(2), "a"),
    ],
)
def test_get_item_works_with_tuple_and_list_indexing_and_str(
    constructor_eager: ConstructorEager,
    row_idx: list[int] | tuple[int] | range,
    col: str,
) -> None:
    nw_df = nw.from_native(constructor_eager(data), eager_only=True)
    nw_df[row_idx, col]
