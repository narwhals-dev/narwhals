from __future__ import annotations

from typing import Any

import pytest

import narwhals as nw
from tests.utils import assert_equal_data

pytest.importorskip("pandas", minversion="3.0.0")
pytest.importorskip("pyarrow")

import numpy as np
import pandas as pd
import pyarrow as pa

STRING_DTYPE_NAN = pd.StringDtype("pyarrow", na_value=np.nan)  # type: ignore[call-arg]
STRING_DTYPE_NA = pd.StringDtype("pyarrow", na_value=pd.NA)  # type: ignore[call-arg]
STRING_DTYPE_PYTHON_NAN = pd.StringDtype("python", na_value=np.nan)  # type: ignore[call-arg]
STRING_DTYPE_PYTHON_NA = pd.StringDtype("python", na_value=pd.NA)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("left_dtype", "right_dtype", "result_dtype"),
    [
        (STRING_DTYPE_NAN, STRING_DTYPE_NAN, STRING_DTYPE_NAN),
        (STRING_DTYPE_NAN, STRING_DTYPE_NA, STRING_DTYPE_NAN),
        (STRING_DTYPE_NAN, pd.ArrowDtype(pa.string()), STRING_DTYPE_NAN),
        (STRING_DTYPE_NAN, pd.ArrowDtype(pa.large_string()), STRING_DTYPE_NAN),
        (STRING_DTYPE_NAN, STRING_DTYPE_PYTHON_NAN, STRING_DTYPE_NAN),
        (STRING_DTYPE_NAN, STRING_DTYPE_PYTHON_NA, STRING_DTYPE_NAN),
        (STRING_DTYPE_NA, STRING_DTYPE_NAN, STRING_DTYPE_NA),
        (STRING_DTYPE_NA, STRING_DTYPE_NA, STRING_DTYPE_NA),
        (STRING_DTYPE_NA, pd.ArrowDtype(pa.string()), STRING_DTYPE_NA),
        (STRING_DTYPE_NA, pd.ArrowDtype(pa.large_string()), STRING_DTYPE_NA),
        (STRING_DTYPE_NA, STRING_DTYPE_PYTHON_NAN, STRING_DTYPE_NA),
        (STRING_DTYPE_NA, STRING_DTYPE_PYTHON_NA, STRING_DTYPE_NA),
        (pd.ArrowDtype(pa.string()), STRING_DTYPE_NAN, pd.ArrowDtype(pa.large_string())),
        (pd.ArrowDtype(pa.string()), STRING_DTYPE_NA, pd.ArrowDtype(pa.large_string())),
        (
            pd.ArrowDtype(pa.string()),
            pd.ArrowDtype(pa.string()),
            pd.ArrowDtype(pa.string()),
        ),
        (
            pd.ArrowDtype(pa.string()),
            pd.ArrowDtype(pa.large_string()),
            pd.ArrowDtype(pa.large_string()),
        ),
        (
            pd.ArrowDtype(pa.large_string()),
            STRING_DTYPE_NAN,
            pd.ArrowDtype(pa.large_string()),
        ),
        (
            pd.ArrowDtype(pa.large_string()),
            STRING_DTYPE_NA,
            pd.ArrowDtype(pa.large_string()),
        ),
        (
            pd.ArrowDtype(pa.large_string()),
            pd.ArrowDtype(pa.string()),
            pd.ArrowDtype(pa.large_string()),
        ),
        (
            pd.ArrowDtype(pa.large_string()),
            pd.ArrowDtype(pa.large_string()),
            pd.ArrowDtype(pa.large_string()),
        ),
        (STRING_DTYPE_PYTHON_NAN, STRING_DTYPE_PYTHON_NAN, STRING_DTYPE_PYTHON_NAN),
        (STRING_DTYPE_PYTHON_NAN, STRING_DTYPE_PYTHON_NA, STRING_DTYPE_PYTHON_NA),
        (STRING_DTYPE_PYTHON_NAN, STRING_DTYPE_NAN, STRING_DTYPE_NAN),
        (STRING_DTYPE_PYTHON_NAN, STRING_DTYPE_NA, STRING_DTYPE_NA),
        (STRING_DTYPE_PYTHON_NAN, pd.ArrowDtype(pa.string()), pd.ArrowDtype(pa.string())),
        (
            STRING_DTYPE_PYTHON_NAN,
            pd.ArrowDtype(pa.large_string()),
            pd.ArrowDtype(pa.large_string()),
        ),
        (STRING_DTYPE_PYTHON_NA, STRING_DTYPE_PYTHON_NAN, STRING_DTYPE_PYTHON_NA),
        (STRING_DTYPE_PYTHON_NA, STRING_DTYPE_PYTHON_NA, STRING_DTYPE_PYTHON_NA),
        (STRING_DTYPE_PYTHON_NA, STRING_DTYPE_NAN, STRING_DTYPE_PYTHON_NA),
        (STRING_DTYPE_PYTHON_NA, STRING_DTYPE_NA, STRING_DTYPE_NA),
        (STRING_DTYPE_PYTHON_NA, pd.ArrowDtype(pa.string()), pd.ArrowDtype(pa.string())),
        (
            STRING_DTYPE_PYTHON_NA,
            pd.ArrowDtype(pa.large_string()),
            pd.ArrowDtype(pa.large_string()),
        ),
        (pd.ArrowDtype(pa.string()), STRING_DTYPE_PYTHON_NAN, pd.ArrowDtype(pa.string())),
        (pd.ArrowDtype(pa.string()), STRING_DTYPE_PYTHON_NA, pd.ArrowDtype(pa.string())),
        (
            pd.ArrowDtype(pa.large_string()),
            STRING_DTYPE_PYTHON_NAN,
            pd.ArrowDtype(pa.large_string()),
        ),
        (
            pd.ArrowDtype(pa.large_string()),
            STRING_DTYPE_PYTHON_NA,
            pd.ArrowDtype(pa.large_string()),
        ),
    ],
)
def test_pandas_str_types(left_dtype: Any, right_dtype: Any, result_dtype: Any) -> None:
    import pandas as pd

    df = pd.DataFrame({"fruit": ["apple", "banana"]}, dtype=left_dtype)
    df["new_str_col"] = "!"
    df["new_str_col"] = df["new_str_col"].astype(right_dtype)  # pyrefly: ignore[missing-attribute] https://github.com/facebook/pyrefly/issues/3299
    res = nw.from_native(df).with_columns(
        concat_col=nw.concat_str([nw.col("fruit"), nw.col("new_str_col")])
    )
    expected = {
        "fruit": ["apple", "banana"],
        "new_str_col": ["!", "!"],
        "concat_col": ["apple!", "banana!"],
    }
    assert_equal_data(res, expected)
    assert res.to_native()["concat_col"].dtype == result_dtype
