from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version
from tests.utils import Constructor
from tests.utils import assert_equal_data


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
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))
    result = df.with_columns(nw.col("a") + 1, d=nw.col("a") - 1)
    assert result.collect_schema().names() == ["a", "b", "z", "d"]
    expected = {"a": [2, 4, 3], "b": [4, 4, 6], "z": [7.0, 8, 9], "d": [0, 2, 1]}
    assert_equal_data(result, expected)


def test_with_columns_empty(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))
    result = df.select().with_columns()
    assert_equal_data(result, {})


def test_with_columns_order_single_row(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9], "i": [0, 1, 2]}
    df = nw.from_native(constructor(data)).filter(nw.col("i") < 1).drop("i")
    result = df.with_columns(nw.col("a") + 1, d=nw.col("a") - 1)
    assert result.collect_schema().names() == ["a", "b", "z", "d"]
    expected = {"a": [2], "b": [4], "z": [7.0], "d": [0]}
    assert_equal_data(result, expected)


def test_with_columns_dtypes_single_row(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "pyarrow_table" in str(constructor) and parse_version(pa.__version__) < (15,):
        request.applymarker(pytest.mark.xfail)
    data = {"a": ["foo"]}
    df = nw.from_native(constructor(data)).with_columns(nw.col("a").cast(nw.Categorical))
    result = df.with_columns(nw.col("a"))
    assert result.collect_schema() == {"a": nw.Categorical}
