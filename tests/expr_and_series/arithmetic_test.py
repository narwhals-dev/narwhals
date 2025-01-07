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
from tests.utils import DASK_VERSION
from tests.utils import PANDAS_VERSION
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


@pytest.mark.parametrize(
    ("attr", "rhs", "expected"),
    [
        ("__add__", 1, [2, 3, 4]),
        ("__sub__", 1, [0, 1, 2]),
        ("__mul__", 2, [2, 4, 6]),
        ("__truediv__", 2.0, [0.5, 1.0, 1.5]),
        ("__truediv__", 1, [1, 2, 3]),
        ("__floordiv__", 2, [0, 1, 1]),
        ("__mod__", 2, [1, 0, 1]),
        ("__pow__", 2, [1, 4, 9]),
    ],
)
def test_arithmetic_expr(
    attr: str,
    rhs: Any,
    expected: list[Any],
    constructor: Constructor,
    request: pytest.FixtureRequest,
) -> None:
    if "duckdb" in str(constructor) and attr == "__floordiv__":
        request.applymarker(pytest.mark.xfail)
    if attr == "__mod__" and any(
        x in str(constructor) for x in ["pandas_pyarrow", "modin_pyarrow"]
    ):
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1.0, 2, 3]}
    df = nw.from_native(constructor(data))
    result = df.select(getattr(nw.col("a"), attr)(rhs))
    assert_equal_data(result, {"a": expected})


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
def test_right_arithmetic_expr(
    attr: str,
    rhs: Any,
    expected: list[Any],
    constructor: Constructor,
    request: pytest.FixtureRequest,
) -> None:
    if "dask" in str(constructor) and DASK_VERSION < (2024, 10):
        request.applymarker(pytest.mark.xfail)
    if attr == "__rmod__" and any(
        x in str(constructor) for x in ["pandas_pyarrow", "modin_pyarrow"]
    ):
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 2, 3]}
    df = nw.from_native(constructor(data))
    result = df.select(getattr(nw.col("a"), attr)(rhs))
    assert_equal_data(result, {"literal": expected})


@pytest.mark.parametrize(
    ("attr", "rhs", "expected"),
    [
        ("__add__", 1, [2, 3, 4]),
        ("__sub__", 1, [0, 1, 2]),
        ("__mul__", 2, [2, 4, 6]),
        ("__truediv__", 2.0, [0.5, 1.0, 1.5]),
        ("__truediv__", 1, [1, 2, 3]),
        ("__floordiv__", 2, [0, 1, 1]),
        ("__mod__", 2, [1, 0, 1]),
        ("__pow__", 2, [1, 4, 9]),
    ],
)
def test_arithmetic_series(
    attr: str,
    rhs: Any,
    expected: list[Any],
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
) -> None:
    if attr == "__mod__" and any(
        x in str(constructor_eager) for x in ["pandas_pyarrow", "modin_pyarrow"]
    ):
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 2, 3]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(getattr(df["a"], attr)(rhs))
    assert_equal_data(result, {"a": expected})


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
def test_right_arithmetic_series(
    attr: str,
    rhs: Any,
    expected: list[Any],
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
) -> None:
    if attr == "__rmod__" and any(
        x in str(constructor_eager) for x in ["pandas_pyarrow", "modin_pyarrow"]
    ):
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 2, 3]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result_series = getattr(df["a"], attr)(rhs)
    assert result_series.name == "a"
    assert_equal_data({"a": result_series}, {"a": expected})


def test_truediv_same_dims(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "polars" in str(constructor_eager):
        # https://github.com/pola-rs/polars/issues/17760
        request.applymarker(pytest.mark.xfail)
    s_left = nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)["a"]
    s_right = nw.from_native(constructor_eager({"a": [2, 2, 1]}), eager_only=True)["a"]
    result = s_left / s_right
    assert_equal_data({"a": result}, {"a": [0.5, 1.0, 3.0]})
    result = s_left.__rtruediv__(s_right)
    assert_equal_data({"a": result}, {"a": [2, 1, 1 / 3]})


@pytest.mark.slow
@given(  # type: ignore[misc]
    left=st.integers(-100, 100),
    right=st.integers(-100, 100),
)
@pytest.mark.skipif(PANDAS_VERSION < (2, 0), reason="convert_dtypes not available")
def test_floordiv(left: int, right: int) -> None:
    # hypothesis complains if we add `constructor` as an argument, so this
    # test is a bit manual unfortunately
    assume(right != 0)
    expected = {"a": [left // right]}
    result = nw.from_native(pd.DataFrame({"a": [left]}), eager_only=True).select(
        nw.col("a") // right
    )
    assert_equal_data(result, expected)
    if PANDAS_VERSION < (2, 2):  # pragma: no cover
        # Bug in old version of pandas
        pass
    else:
        result = nw.from_native(
            pd.DataFrame({"a": [left]}).convert_dtypes(dtype_backend="pyarrow"),
            eager_only=True,
        ).select(nw.col("a") // right)
        assert_equal_data(result, expected)
    result = nw.from_native(
        pd.DataFrame({"a": [left]}).convert_dtypes(), eager_only=True
    ).select(nw.col("a") // right)
    assert_equal_data(result, expected)
    result = nw.from_native(pl.DataFrame({"a": [left]}), eager_only=True).select(
        nw.col("a") // right
    )
    assert_equal_data(result, expected)
    result = nw.from_native(pa.table({"a": [left]}), eager_only=True).select(
        nw.col("a") // right
    )
    assert_equal_data(result, expected)


@pytest.mark.slow
@given(  # type: ignore[misc]
    left=st.integers(-100, 100),
    right=st.integers(-100, 100),
)
@pytest.mark.skipif(PANDAS_VERSION < (2, 0), reason="convert_dtypes not available")
def test_mod(left: int, right: int) -> None:
    # hypothesis complains if we add `constructor` as an argument, so this
    # test is a bit manual unfortunately
    assume(right != 0)
    expected = {"a": [left % right]}
    result = nw.from_native(pd.DataFrame({"a": [left]}), eager_only=True).select(
        nw.col("a") % right
    )
    assert_equal_data(result, expected)
    result = nw.from_native(
        pd.DataFrame({"a": [left]}).convert_dtypes(), eager_only=True
    ).select(nw.col("a") % right)
    assert_equal_data(result, expected)
    result = nw.from_native(pl.DataFrame({"a": [left]}), eager_only=True).select(
        nw.col("a") % right
    )
    assert_equal_data(result, expected)
    result = nw.from_native(pa.table({"a": [left]}), eager_only=True).select(
        nw.col("a") % right
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("attr", "lhs", "expected"),
    [
        ("__add__", nw.lit(1), [2, 3, 5]),
        ("__sub__", nw.lit(1), [0, -1, -3]),
        ("__mul__", nw.lit(2), [2, 4, 8]),
        ("__truediv__", nw.lit(2.0), [2, 1, 0.5]),
        ("__truediv__", nw.lit(1), [1, 0.5, 0.25]),
        ("__floordiv__", nw.lit(2), [2, 1, 0]),
        ("__mod__", nw.lit(3), [0, 1, 3]),
        ("__pow__", nw.lit(2), [2, 4, 16]),
    ],
)
def test_arithmetic_expr_left_literal(
    attr: str,
    lhs: Any,
    expected: list[Any],
    constructor: Constructor,
    request: pytest.FixtureRequest,
) -> None:
    if ("duckdb" in str(constructor) and attr == "__floordiv__") or (
        "dask" in str(constructor) and DASK_VERSION < (2024, 10)
    ):
        request.applymarker(pytest.mark.xfail)
    if attr == "__mod__" and any(
        x in str(constructor) for x in ["pandas_pyarrow", "modin_pyarrow"]
    ):
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1.0, 2, 4]}
    df = nw.from_native(constructor(data))
    result = df.select(getattr(lhs, attr)(nw.col("a")))
    assert_equal_data(result, {"literal": expected})


@pytest.mark.parametrize(
    ("attr", "lhs", "expected"),
    [
        ("__add__", nw.lit(1), [2, 3, 5]),
        ("__sub__", nw.lit(1), [0, -1, -3]),
        ("__mul__", nw.lit(2), [2, 4, 8]),
        ("__truediv__", nw.lit(2.0), [2, 1, 0.5]),
        ("__truediv__", nw.lit(1), [1, 0.5, 0.25]),
        ("__floordiv__", nw.lit(2), [2, 1, 0]),
        ("__mod__", nw.lit(3), [0, 1, 3]),
        ("__pow__", nw.lit(2), [2, 4, 16]),
    ],
)
def test_arithmetic_series_left_literal(
    attr: str,
    lhs: Any,
    expected: list[Any],
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
) -> None:
    if attr == "__mod__" and any(
        x in str(constructor_eager) for x in ["pandas_pyarrow", "modin_pyarrow"]
    ):
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1.0, 2, 4]}
    df = nw.from_native(constructor_eager(data))
    result = df.select(getattr(lhs, attr)(nw.col("a")))
    assert_equal_data(result, {"literal": expected})
