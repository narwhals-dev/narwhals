from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pytest

import narwhals.stable.v1 as nw
from tests.utils import PANDAS_VERSION
from tests.utils import Constructor
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from narwhals.dtypes import DType


@pytest.mark.parametrize(
    ("dtype", "expected_lit"),
    [(None, [2, 2, 2]), (nw.String, ["2", "2", "2"]), (nw.Float32, [2.0, 2.0, 2.0])],
)
def test_lit(
    constructor: Constructor,
    dtype: DType | None,
    expected_lit: list[Any],
) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw).lazy()
    result = df.with_columns(nw.lit(2, dtype).alias("lit"))
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "z": [7.0, 8.0, 9.0],
        "lit": expected_lit,
    }
    assert_equal_data(result, expected)


def test_lit_error(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw).lazy()
    with pytest.raises(
        ValueError, match="numpy arrays are not supported as literal values"
    ):
        _ = df.with_columns(nw.lit(np.array([1, 2])).alias("lit"))
    with pytest.raises(
        NotImplementedError, match="Nested datatypes are not supported yet."
    ):
        _ = df.with_columns(nw.lit((1, 2)).alias("lit"))
    with pytest.raises(
        NotImplementedError, match="Nested datatypes are not supported yet."
    ):
        _ = df.with_columns(nw.lit([1, 2]).alias("lit"))


def test_lit_out_name(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw).lazy()
    result = df.with_columns(nw.lit(2))
    expected = {
        "a": [1, 3, 2],
        "literal": [2, 2, 2],
    }
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("col_name", "expr", "expected_result"),
    [
        ("left_lit", nw.lit(1) + nw.col("a"), [2, 4, 3]),
        ("right_lit", nw.col("a") + nw.lit(1), [2, 4, 3]),
        ("left_lit_with_agg", nw.lit(1) + nw.col("a").mean(), [3]),
        ("right_lit_with_agg", nw.col("a").mean() - nw.lit(1), [1]),
        ("left_scalar", 1 + nw.col("a"), [2, 4, 3]),
        ("right_scalar", nw.col("a") + 1, [2, 4, 3]),
        ("left_scalar_with_agg", 1 + nw.col("a").mean(), [3]),
        ("right_scalar_with_agg", nw.col("a").mean() - 1, [1]),
        ("lit_compare", nw.col("a") == nw.lit(3), [False, True, False]),
    ],
)
def test_lit_operation_in_select(
    constructor: Constructor,
    col_name: str,
    expr: nw.Expr,
    expected_result: list[int],
) -> None:
    data = {"a": [1, 3, 2]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw).lazy()
    result = df.select(expr.alias(col_name))
    expected = {col_name: expected_result}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("col_name", "expr", "expected_result"),
    [
        ("lit_and_scalar", (nw.lit(2) + 1), [3, 3, 3]),
        ("scalar_and_lit", (1 + nw.lit(2)), [3, 3, 3]),
    ],
)
def test_lit_operation_in_with_columns(
    constructor: Constructor,
    col_name: str,
    expr: nw.Expr,
    expected_result: list[int],
) -> None:
    data = {"a": [1, 3, 2]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw).lazy()
    result = df.with_columns(expr.alias(col_name))
    expected = {"a": data["a"], col_name: expected_result}
    assert_equal_data(result, expected)


@pytest.mark.skipif(PANDAS_VERSION < (1, 5), reason="too old for pyarrow")
def test_date_lit(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "dask" in str(constructor):
        # https://github.com/dask/dask/issues/11637
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor({"a": [1]}))
    result = df.with_columns(nw.lit(date(2020, 1, 1), dtype=nw.Date)).collect_schema()
    if df.implementation.is_cudf():
        # cudf has no date dtype
        assert result == {"a": nw.Int64, "literal": nw.Datetime}
    else:
        assert result == {"a": nw.Int64, "literal": nw.Date}
