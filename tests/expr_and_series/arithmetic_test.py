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


@pytest.mark.parametrize(
    ("attr", "rhs", "expected"),
    [
        ("__add__", 1, [2, 3, -2]),
        ("__sub__", 1, [0, 1, -4]),
        ("__mul__", 2, [2, 4, -6]),
        ("__truediv__", 2.0, [0.5, 1.0, -1.5]),
        ("__floordiv__", 2, [0, 1, -2]),
        ("__mod__", 2, [1, 0, 1]),
        ("__mod__", 4, [1, 2, 1]),
        ("__mod__", -4, [-3, -2, -3]),
        ("__pow__", 2, [1, 4, 9]),
    ],
)
@pytest.mark.parametrize("data", [[1, 2, -3], [1.0, 2, -3]])
def test_arithmetic(
    request: Any,
    constructor: Any,
    attr: str,
    rhs: Any,
    expected: list[Any],
    data: list[float],
) -> None:
    if "pandas_pyarrow" in str(constructor) and attr == "__mod__":
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor({"a": data}))
    result_expr = df.select(getattr(nw.col("a"), attr)(rhs))
    compare_dicts(result_expr, {"a": expected})

    series = df["a"]
    result_series = getattr(series, attr)(rhs)
    assert result_series.to_numpy().tolist() == expected


@pytest.mark.parametrize(
    ("attr", "rhs", "expected"),
    [
        ("__radd__", 1, [2, 3, 4]),
        ("__rsub__", 1, [0, -1, -2]),
        ("__rmul__", 2, [2, 4, 6]),
        ("__rtruediv__", 2.0, [2, 1, 2 / 3]),
        ("__rfloordiv__", 2, [2, 1, 0]),
        ("__rmod__", 2, [0, 0, 2]),
        ("__rpow__", 2, [2, 4, 8]),
    ],
)
def test_right_arithmetic(
    attr: str, rhs: Any, expected: list[Any], constructor: Any, request: Any
) -> None:
    if "pandas_pyarrow" in str(constructor) and attr in {"__rmod__"}:
        request.applymarker(pytest.mark.xfail)

    # pyarrow case
    if "table" in str(constructor) and attr in {"__rmod__"}:
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 2, 3]}
    df = nw.from_native(constructor(data))
    result_expr = df.select(a=getattr(nw.col("a"), attr)(rhs))
    compare_dicts(result_expr, {"a": expected})

    series = df["a"]
    result_series = getattr(series, attr)(rhs)
    assert result_series.to_numpy().tolist() == expected


def test_truediv_same_dims(constructor_series: Any, request: Any) -> None:
    if "polars" in str(constructor_series):
        # https://github.com/pola-rs/polars/issues/17760
        request.applymarker(pytest.mark.xfail)
    s_left = nw.from_native(constructor_series([1, 2, 3]), series_only=True)
    s_right = nw.from_native(constructor_series([2, 2, 1]), series_only=True)
    result = (s_left / s_right).to_list()
    assert result == [0.5, 1.0, 3.0]
    result = (s_left.__rtruediv__(s_right)).to_list()
    assert result == [2, 1, 1 / 3]


@pytest.mark.slow()
@given(  # type: ignore[misc]
    left=st.integers(-100, 100),
    right=st.integers(-100, 100),
)
@pytest.mark.skipif(
    parse_version(pd.__version__) < (2, 0), reason="convert_dtypes not available"
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
    if parse_version(pd.__version__) < (2, 2):  # pragma: no cover
        # Bug in old version of pandas
        pass
    else:
        result = nw.from_native(
            pd.DataFrame({"a": [left]}).convert_dtypes(dtype_backend="pyarrow"),
            eager_only=True,
        ).select(nw.col("a") // right)
        compare_dicts(result, expected)
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
