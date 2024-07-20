from __future__ import annotations

from typing import Any

import hypothesis.strategies as st
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from hypothesis import assume
from hypothesis import given

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version
from tests.utils import compare_dicts


@pytest.mark.parametrize("data", [[1, 2, 3], [1.0, 2, 3]])
@pytest.mark.parametrize(
    ("attr", "rhs", "expected"),
    [
        ("__add__", 1, [2, 3, 4]),
        ("__sub__", 1, [0, 1, 2]),
        ("__mul__", 2, [2, 4, 6]),
        ("__truediv__", 2, [0.5, 1.0, 1.5]),
        ("__truediv__", 1, [1, 2, 3]),
        ("__floordiv__", 2, [0, 1, 1]),
        ("__mod__", 2, [1, 0, 1]),
        ("__pow__", 2, [1, 4, 9]),
    ],
)
def test_arithmetic(
    request: Any,
    data: list[int | float],
    attr: str,
    rhs: Any,
    expected: list[Any],
    constructor_series: Any,
) -> None:
    if "pandas_series_pyarrow" in str(constructor_series) and attr == "__mod__":
        request.applymarker(pytest.mark.xfail)

    if "pyarrow_series" in str(constructor_series) and attr in {
        "__truediv__",
        "__mod__",
    }:
        request.applymarker(pytest.mark.xfail)

    s = nw.from_native(constructor_series(data), series_only=True)
    result = getattr(s, attr)(rhs)
    assert result.to_numpy().tolist() == expected


@pytest.mark.slow()
@given(  # type: ignore[misc]
    left=st.integers(-100, 100),
    right=st.integers(-100, 100),
)
def test_mod(left: int, right: int) -> None:
    # hypothesis complains if we add `constructor` as an argument, so this
    # test is a bit manual unfortunately
    assume(right != 0)
    expected = {"a": [left // right]}
    result = nw.from_native(pd.DataFrame({"a": [left]}), eager_only=True).select(
        nw.col("a") // right
    )
    compare_dicts(result, expected)
    if parse_version(pd.__version__) >= (2, 2):
        # Bug in old version of pandas
        result = nw.from_native(
            pd.DataFrame({"a": [left]}).convert_dtypes(dtype_backend="pyarrow"),
            eager_only=True,
        ).select(nw.col("a") // right)
        compare_dicts(result, expected)
    else:  # pragma: no cover
        pass
    result = nw.from_native(
        pd.DataFrame({"a": [left]}).convert_dtypes(), eager_only=True
    ).select(nw.col("a") // right)
    compare_dicts(result, expected)
    result = nw.from_native(pl.DataFrame({"a": [left]}), eager_only=True).select(
        nw.col("a") // right
    )
    compare_dicts(result, expected)
    result = nw.from_native(pa.table({"a": [left]}), eager_only=True).select(
        nw.col("a") // right
    )
    compare_dicts(result, expected)
