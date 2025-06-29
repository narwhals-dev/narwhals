from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import POLARS_VERSION, Constructor, ConstructorEager, assert_equal_data

data = {"int": [-2, 0, 2], "float": [-2.1, 0.0, 2.1]}
expected_truediv = [float("-inf"), float("nan"), float("inf")]
expected_floordiv = [None, None, None]


def test_series_truediv_by_zero(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = {"int": df["int"] / 0, "float": df["float"] / 0}
    expected = {"int": expected_truediv, "float": expected_truediv}
    assert_equal_data(result, expected)


def test_expr_truediv_by_zero(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.select(nw.all() / 0)
    expected = {"int": expected_truediv, "float": expected_truediv}
    assert_equal_data(result, expected)


def test_series_floordiv_by_zero(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "polars" in str(constructor_eager) and POLARS_VERSION < (0, 20, 6):
        pytest.skip(reason="bug")
    df = nw.from_native(constructor_eager(data), eager_only=True)

    if df.implementation.is_pandas_like():
        reason = "floordiv either fails or generate different result"
        request.applymarker(pytest.mark.xfail(reason=reason))

    result = {"int": df["int"] // 0}
    expected = {"int": expected_floordiv}
    assert_equal_data(result, expected)


def test_expr_floordiv_by_zero(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (0, 20, 6):
        pytest.skip(reason="bug")

    df = nw.from_native(constructor(data))

    if df.implementation.is_pandas_like():
        reason = "floordiv either fails or generate different result"
        request.applymarker(pytest.mark.xfail(reason=reason))

    result = df.select(nw.col("int") // 0)
    expected = {"int": expected_floordiv}
    assert_equal_data(result, expected)
