from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import narwhals as nw
from narwhals.exceptions import ShapeError
from tests.utils import PYARROW_VERSION, Constructor, ConstructorEager, assert_equal_data


def test_with_columns_int_col_name_pandas() -> None:
    np_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    df = pd.DataFrame(np_matrix, dtype="int64")
    nw_df = nw.from_native(df, eager_only=True)
    result = nw_df.with_columns(nw_df.get_column(1).alias(4)).pipe(nw.to_native)  # type: ignore[arg-type]
    expected = pd.DataFrame(
        {0: [1, 4, 7], 1: [2, 5, 8], 2: [3, 6, 9], 4: [2, 5, 8]}, dtype="int64"
    )
    pd.testing.assert_frame_equal(result, expected)


def test_with_columns_order(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))
    result = df.with_columns(nw.col("a") + 1, d=nw.col("a") - 1)
    assert result.collect_schema().names() == ["a", "b", "z", "d"]
    expected = {"a": [2, 4, 3], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0], "d": [0, 2, 1]}
    assert_equal_data(result, expected)


def test_with_columns_empty(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor_eager(data))
    result = df.select().with_columns()
    assert_equal_data(result, {})


def test_select_with_columns_empty_lazy(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data)).lazy()
    with pytest.raises(ValueError, match="At least one"):
        df.with_columns()
    with pytest.raises(ValueError, match="At least one"):
        df.select()


def test_with_columns_order_single_row(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0], "i": [0, 1, 2]}
    df = nw.from_native(constructor(data)).filter(nw.col("i") < 1).drop("i")
    result = df.with_columns(nw.col("a") + 1, d=nw.col("a") - 1)
    assert result.collect_schema().names() == ["a", "b", "z", "d"]
    expected = {"a": [2], "b": [4], "z": [7.0], "d": [0]}
    assert_equal_data(result, expected)


def test_with_columns_dtypes_single_row(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "pyarrow_table" in str(constructor) and PYARROW_VERSION < (15,):
        pytest.skip()
    if (
        ("pyspark" in str(constructor))
        or "duckdb" in str(constructor)
        or "ibis" in str(constructor)
    ):
        request.applymarker(pytest.mark.xfail)
    data = {"a": ["foo"]}
    df = nw.from_native(constructor(data)).with_columns(nw.col("a").cast(nw.Categorical))
    result = df.with_columns(nw.col("a"))
    assert result.collect_schema() == {"a": nw.Categorical}


def test_with_columns_series_shape_mismatch(constructor_eager: ConstructorEager) -> None:
    df1 = nw.from_native(constructor_eager({"first": [1, 2, 3]}), eager_only=True)
    second = nw.from_native(constructor_eager({"second": [1, 2, 3, 4]}), eager_only=True)[
        "second"
    ]
    with pytest.raises(ShapeError):
        df1.with_columns(second=second)
