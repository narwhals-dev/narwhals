from __future__ import annotations

from typing import Any

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given

import narwhals as nw
from tests.utils import (
    DASK_VERSION,
    DUCKDB_VERSION,
    PANDAS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)


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

    data = {"a": [1.0, 2.0, 3.0]}
    df = nw.from_native(constructor(data))
    result = df.select(getattr(nw.col("a"), attr)(rhs))
    assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(
    ("attr", "rhs", "expected"),
    [
        ("__radd__", 1, [2, 3, 4]),
        ("__rsub__", 1, [0, -1, -2]),
        ("__rmul__", 2, [2, 4, 6]),
        ("__rtruediv__", 2.0, [2.0, 1.0, 2 / 3]),
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
        pytest.skip()
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
        ("__rtruediv__", 2.0, [2.0, 1.0, 2 / 3]),
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


@given(left=st.integers(-100, 100), right=st.integers(-100, 100))
@pytest.mark.skipif(PANDAS_VERSION < (2, 0), reason="convert_dtypes not available")
@pytest.mark.slow
def test_floordiv(constructor_eager: ConstructorEager, *, left: int, right: int) -> None:
    if any(x in str(constructor_eager) for x in ["modin", "cudf"]):
        # modin & cudf are too slow here
        pytest.skip()
    assume(right != 0)
    expected = {"a": [left // right]}
    result = nw.from_native(constructor_eager({"a": [left]}), eager_only=True).select(
        nw.col("a") // right
    )
    assert_equal_data(result, expected)


@pytest.mark.slow
@given(left=st.integers(-100, 100), right=st.integers(-100, 100))
@pytest.mark.skipif(PANDAS_VERSION < (2, 0), reason="convert_dtypes not available")
def test_mod(constructor_eager: ConstructorEager, *, left: int, right: int) -> None:
    if any(x in str(constructor_eager) for x in ["pandas_pyarrow", "modin", "cudf"]):
        # pandas[pyarrow] does not implement mod
        # modin & cudf are too slow here
        pytest.skip()
    assume(right != 0)
    expected = {"a": [left % right]}
    result = nw.from_native(constructor_eager({"a": [left]}), eager_only=True).select(
        nw.col("a") % right
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("attr", "lhs", "expected"),
    [
        ("__add__", nw.lit(1), [2, 3, 5]),
        ("__sub__", nw.lit(1), [0, -1, -3]),
        ("__mul__", nw.lit(2), [2, 4, 8]),
        ("__truediv__", nw.lit(2.0), [2.0, 1.0, 0.5]),
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

    data = {"a": [1.0, 2.0, 4.0]}
    df = nw.from_native(constructor(data))
    result = df.select(getattr(lhs, attr)(nw.col("a")))
    assert_equal_data(result, {"literal": expected})


@pytest.mark.parametrize(
    ("attr", "lhs", "expected"),
    [
        ("__add__", nw.lit(1), [2, 3, 5]),
        ("__sub__", nw.lit(1), [0, -1, -3]),
        ("__mul__", nw.lit(2), [2, 4, 8]),
        ("__truediv__", nw.lit(2.0), [2.0, 1.0, 0.5]),
        ("__truediv__", nw.lit(1), [1.0, 0.5, 0.25]),
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

    data = {"a": [1.0, 2.0, 4.0]}
    df = nw.from_native(constructor_eager(data))
    result = df.select(getattr(lhs, attr)(nw.col("a")))
    assert_equal_data(result, {"literal": expected})


def test_std_broadcating(constructor: Constructor) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        # `std(ddof=2)` fails for duckdb here
        pytest.skip()
    df = nw.from_native(constructor({"a": [1, 2, 3]}))
    result = df.with_columns(b=nw.col("a").std()).sort("a")
    expected = {"a": [1, 2, 3], "b": [1.0, 1.0, 1.0]}
    assert_equal_data(result, expected)
    result = df.with_columns(b=nw.col("a").var()).sort("a")
    expected = {"a": [1, 2, 3], "b": [1.0, 1.0, 1.0]}
    assert_equal_data(result, expected)
    result = df.with_columns(b=nw.col("a").std(ddof=0)).sort("a")
    expected = {
        "a": [1, 2, 3],
        "b": [0.816496580927726, 0.816496580927726, 0.816496580927726],
    }
    assert_equal_data(result, expected)
    result = df.with_columns(b=nw.col("a").var(ddof=0)).sort("a")
    expected = {
        "a": [1, 2, 3],
        "b": [0.6666666666666666, 0.6666666666666666, 0.6666666666666666],
    }
    assert_equal_data(result, expected)
