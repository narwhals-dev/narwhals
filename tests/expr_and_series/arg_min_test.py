from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import POLARS_VERSION, ConstructorEager, assert_equal_data

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}


def test_expr_arg_min_expr(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "cudf" in str(constructor_eager):
        # not implemented yet
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.col("a", "b", "z").arg_min())
    expected = {"a": [0], "b": [0], "z": [0]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(("col", "expected"), [("a", 0), ("b", 0), ("z", 0)])
def test_expr_arg_min_series(
    constructor_eager: ConstructorEager,
    col: str,
    expected: float,
    request: pytest.FixtureRequest,
) -> None:
    if "modin" in str(constructor_eager):
        # TODO(unassigned): bug in modin?
        return
    if "cudf" in str(constructor_eager):
        # not implemented yet
        request.applymarker(pytest.mark.xfail)
    series = nw.from_native(constructor_eager(data), eager_only=True)[col]
    series = nw.maybe_set_index(series, index=[1, 0, 9])  # type: ignore[arg-type]
    result = series.arg_min()
    assert_equal_data({col: [result]}, {col: [expected]})


def test_expr_arg_min_over() -> None:
    # This is tricky. But, we may be able to support it for
    # other backends too one day.
    pytest.importorskip("polars")
    import polars as pl

    if POLARS_VERSION < (1, 10):
        pytest.skip()

    df = nw.from_native(pl.LazyFrame({"a": [9, 8, 7], "i": [0, 2, 1]}))
    result = df.select(nw.col("a").arg_min().over(order_by="i"))
    expected = {"a": [1, 1, 1]}
    assert_equal_data(result, expected)
