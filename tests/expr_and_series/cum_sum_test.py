from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import (
    DUCKDB_VERSION,
    POLARS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

data = {"arg entina": [1, 2, None, 4]}
expected = {"cum_sum": [1, 3, None, 7], "reverse_cum_sum": [7, 6, None, 4]}


@pytest.mark.parametrize("reverse", [True, False])
def test_cum_sum_expr(constructor_eager: ConstructorEager, *, reverse: bool) -> None:
    name = "reverse_cum_sum" if reverse else "cum_sum"
    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.col("arg entina").cum_sum(reverse=reverse).alias(name))

    assert_equal_data(result, {name: expected[name]})


@pytest.mark.parametrize(
    ("reverse", "expected_a"), [(False, [3, 2, 6]), (True, [4, 6, 3])]
)
def test_lazy_cum_sum_grouped(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    *,
    reverse: bool,
    expected_a: list[int],
) -> None:
    if "pyarrow_table" in str(constructor):
        # grouped window functions not yet supported
        request.applymarker(pytest.mark.xfail)
    if "dask" in str(constructor):
        # https://github.com/dask/dask/issues/11806
        request.applymarker(pytest.mark.xfail)
    if ("polars" in str(constructor) and POLARS_VERSION < (1, 9)) or (
        "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3)
    ):
        pytest.skip(reason="too old version")
    if "cudf" in str(constructor):
        # https://github.com/rapidsai/cudf/issues/18159
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(
        constructor(
            {
                "arg entina": [1, 2, 3],
                "ban gkok": [1, 0, 2],
                "i ran": [0, 1, 2],
                "g": [1, 1, 1],
            }
        )
    )
    result = df.with_columns(
        nw.col("arg entina").cum_sum(reverse=reverse).over("g", order_by="ban gkok")
    ).sort("i ran")
    expected = {
        "arg entina": expected_a,
        "ban gkok": [1, 0, 2],
        "i ran": [0, 1, 2],
        "g": [1, 1, 1],
    }
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("reverse", "expected_a"),
    [(False, [10, 6, 14, 11, 16, 9, 4]), (True, [7, 12, 5, 6, 2, 10, 16])],
)
def test_lazy_cum_sum_ordered_by_nulls(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    *,
    reverse: bool,
    expected_a: list[int],
) -> None:
    if "pyarrow_table" in str(constructor):
        # grouped window functions not yet supported
        request.applymarker(pytest.mark.xfail)
    if "dask" in str(constructor):
        # https://github.com/dask/dask/issues/11806
        request.applymarker(pytest.mark.xfail)
    if ("polars" in str(constructor) and POLARS_VERSION < (1, 9)) or (
        "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3)
    ):
        pytest.skip(reason="too old version")
    if "cudf" in str(constructor):
        # https://github.com/rapidsai/cudf/issues/18159
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(
        constructor(
            {
                "arg entina": [1, 2, 3, 1, 2, 3, 4],
                "ban gkok": [1, -1, 3, 2, 5, 0, None],
                "i ran": [0, 1, 2, 3, 4, 5, 6],
                "g": [1, 1, 1, 1, 1, 1, 1],
            }
        )
    )
    result = df.with_columns(
        nw.col("arg entina").cum_sum(reverse=reverse).over("g", order_by="ban gkok")
    ).sort("i ran")
    expected = {
        "arg entina": expected_a,
        "ban gkok": [1, -1, 3, 2, 5, 0, None],
        "i ran": [0, 1, 2, 3, 4, 5, 6],
        "g": [1, 1, 1, 1, 1, 1, 1],
    }
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("reverse", "expected_a"), [(False, [3, 2, 6]), (True, [4, 6, 3])]
)
def test_lazy_cum_sum_ungrouped(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    *,
    reverse: bool,
    expected_a: list[int],
) -> None:
    if "dask" in str(constructor) and reverse:
        # https://github.com/dask/dask/issues/11802
        request.applymarker(pytest.mark.xfail)
    if ("polars" in str(constructor) and POLARS_VERSION < (1, 9)) or (
        "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3)
    ):
        pytest.skip(reason="too old version")

    df = nw.from_native(
        constructor({"arg entina": [2, 3, 1], "ban gkok": [0, 2, 1], "i ran": [1, 2, 0]})
    ).sort("i ran")
    result = df.with_columns(
        nw.col("arg entina").cum_sum(reverse=reverse).over(order_by="ban gkok")
    ).sort("i ran")
    expected = {"arg entina": expected_a, "ban gkok": [1, 0, 2], "i ran": [0, 1, 2]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("reverse", "expected_a"),
    [(False, [10, 6, 14, 11, 16, 9, 4]), (True, [7, 12, 5, 6, 2, 10, 16])],
)
def test_lazy_cum_sum_ungrouped_ordered_by_nulls(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    *,
    reverse: bool,
    expected_a: list[int],
) -> None:
    if "dask" in str(constructor):
        # https://github.com/dask/dask/issues/11806
        request.applymarker(pytest.mark.xfail)
    if ("polars" in str(constructor) and POLARS_VERSION < (1, 9)) or (
        "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3)
    ):
        pytest.skip(reason="too old version")

    df = nw.from_native(
        constructor(
            {
                "arg entina": [1, 2, 3, 1, 2, 3, 4],
                "ban gkok": [1, -1, 3, 2, 5, 0, None],
                "i ran": [0, 1, 2, 3, 4, 5, 6],
            }
        )
    ).sort("i ran")
    result = df.with_columns(
        nw.col("arg entina").cum_sum(reverse=reverse).over(order_by="ban gkok")
    ).sort("i ran")
    expected = {
        "arg entina": expected_a,
        "ban gkok": [1, -1, 3, 2, 5, 0, None],
        "i ran": [0, 1, 2, 3, 4, 5, 6],
    }
    assert_equal_data(result, expected)


def test_cum_sum_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(
        cum_sum=df["arg entina"].cum_sum(),
        reverse_cum_sum=df["arg entina"].cum_sum(reverse=True),
    )
    assert_equal_data(result, expected)


def test_shift_cum_sum(constructor_eager: ConstructorEager) -> None:
    if "polars" in str(constructor_eager) and POLARS_VERSION < (1, 10):
        pytest.skip()
    data = {"arg entina": [1, 2, 3, 4, 5], "i": list(range(5))}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.with_columns(kalimantan=nw.col("arg entina").shift(1).cum_sum())
    expected = {
        "arg entina": [1, 2, 3, 4, 5],
        "i": list(range(5)),
        "kalimantan": [None, 1, 3, 6, 10],
    }
    assert_equal_data(result, expected)
    result = df.with_columns(
        kalimantan=nw.col("arg entina").shift(1).cum_sum().over(order_by="i")
    )
    expected = {
        "arg entina": [1, 2, 3, 4, 5],
        "i": list(range(5)),
        "kalimantan": [None, 1, 3, 6, 10],
    }
    assert_equal_data(result, expected)
