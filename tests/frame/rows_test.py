from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version

df_pandas = pd.DataFrame({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})
df_pa = pa.table({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})
if parse_version(pd.__version__) >= parse_version("1.5.0"):
    df_pandas_pyarrow = pd.DataFrame(
        {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    ).astype(
        {
            "a": "Int64[pyarrow]",
            "b": "Int64[pyarrow]",
            "z": "Float64[pyarrow]",
        }
    )
    df_pandas_nullable = pd.DataFrame(
        {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    ).astype(
        {
            "a": "Int64",
            "b": "Int64",
            "z": "Float64",
        }
    )
else:  # pragma: no cover
    df_pandas_pyarrow = df_pandas
    df_pandas_nullable = df_pandas
df_polars = pl.DataFrame({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]})

df_pandas_na = pd.DataFrame({"a": [None, 3, 2], "b": [4, 4, 6], "z": [7.0, None, 9]})
df_polars_na = pl.DataFrame({"a": [None, 3, 2], "b": [4, 4, 6], "z": [7.0, None, 9]})


@pytest.mark.parametrize(
    ("named", "expected"),
    [
        (False, [(1, 4, 7.0, 5), (3, 4, 8.0, 6), (2, 6, 9.0, 7)]),
        (
            True,
            [
                {"a": 1, "_b": 4, "z": 7.0, "1": 5},
                {"a": 3, "_b": 4, "z": 8.0, "1": 6},
                {"a": 2, "_b": 6, "z": 9.0, "1": 7},
            ],
        ),
    ],
)
def test_iter_rows(
    request: Any,
    constructor_eager: Any,
    named: bool,  # noqa: FBT001
    expected: list[tuple[Any, ...]] | list[dict[str, Any]],
) -> None:
    if "cudf" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 3, 2], "_b": [4, 4, 6], "z": [7.0, 8, 9], "1": [5, 6, 7]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = list(df.iter_rows(named=named))
    assert result == expected


@pytest.mark.parametrize(
    "df_raw", [df_pandas, df_pandas_nullable, df_pandas_pyarrow, df_polars, df_pa]
)
@pytest.mark.parametrize(
    ("named", "expected"),
    [
        (False, [(1, 4, 7.0), (3, 4, 8.0), (2, 6, 9.0)]),
        (
            True,
            [
                {"a": 1, "b": 4, "z": 7.0},
                {"a": 3, "b": 4, "z": 8.0},
                {"a": 2, "b": 6, "z": 9.0},
            ],
        ),
    ],
)
def test_rows(
    df_raw: Any,
    named: bool,  # noqa: FBT001
    expected: list[tuple[Any, ...]] | list[dict[str, Any]],
) -> None:
    df = nw.from_native(df_raw, eager_only=True)
    result = df.rows(named=named)
    assert result == expected


@pytest.mark.parametrize("df_raw", [df_pandas_na, df_polars_na])
def test_rows_with_nulls_unnamed(df_raw: Any) -> None:
    # GIVEN
    df = nw.from_native(df_raw, eager_only=True)

    # WHEN
    result = list(df.iter_rows(named=False))

    # THEN
    expected = [(None, 4, 7.0), (3, 4, None), (2, 6, 9.0)]
    for i, row in enumerate(expected):
        for j, value in enumerate(row):
            value_in_result = result[i][j]
            if value is None:
                assert pd.isna(value_in_result)  # because float('nan') != float('nan')
            else:
                assert value_in_result == value


@pytest.mark.parametrize("df_raw", [df_pandas_na, df_polars_na])
def test_rows_with_nulls_named(df_raw: Any) -> None:
    # GIVEN
    df = nw.from_native(df_raw, eager_only=True)

    # WHEN
    result = list(df.iter_rows(named=True))

    # THEN
    expected: list[dict[str, Any]] = [
        {"a": None, "b": 4, "z": 7.0},
        {"a": 3, "b": 4, "z": None},
        {"a": 2, "b": 6, "z": 9.0},
    ]
    for i, row in enumerate(expected):
        for col, value in row.items():
            value_in_result = result[i][col]
            if value is None:
                assert pd.isna(value_in_result)  # because float('nan') != float('nan')
            else:
                assert value_in_result == value
