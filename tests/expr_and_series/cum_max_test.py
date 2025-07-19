from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import (
    DUCKDB_VERSION,
    PANDAS_VERSION,
    POLARS_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

data = {"a": [1, 3, None, 2]}

expected = {"cum_max": [1, 3, None, 3], "reverse_cum_max": [3, 3, None, 2]}


@pytest.mark.parametrize("reverse", [True, False])
def test_cum_max_expr(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager, *, reverse: bool
) -> None:
    if (PANDAS_VERSION < (2, 1)) and "pandas_pyarrow" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    name = "reverse_cum_max" if reverse else "cum_max"
    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.col("a").cum_max(reverse=reverse).alias(name))

    assert_equal_data(result, {name: expected[name]})


@pytest.mark.parametrize(
    ("reverse", "expected_a"), [(False, [2, 2, 3]), (True, [3, 3, 3])]
)
def test_lazy_cum_max_grouped(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    *,
    reverse: bool,
    expected_a: list[int],
) -> None:
    if "pyarrow_table" in str(constructor):
        # grouped window functions not yet supported
        request.applymarker(pytest.mark.xfail)
    if "modin" in str(constructor):
        pytest.skip(reason="probably bugged")
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
        constructor({"a": [1, 2, 3], "b": [1, 0, 2], "i": [0, 1, 2], "g": [1, 1, 1]})
    )
    result = df.with_columns(
        nw.col("a").cum_max(reverse=reverse).over("g", order_by="b")
    ).sort("i")
    expected = {"a": expected_a, "b": [1, 0, 2], "i": [0, 1, 2], "g": [1, 1, 1]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("reverse", "expected_a"),
    [(False, [4, 4, 4, 4, 4, 4, 4]), (True, [3, 3, 3, 3, 2, 3, 4])],
)
def test_lazy_cum_max_ordered_by_nulls(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    *,
    reverse: bool,
    expected_a: list[int],
) -> None:
    if "pyarrow_table" in str(constructor):
        # grouped window functions not yet supported
        request.applymarker(pytest.mark.xfail)
    if "modin" in str(constructor):
        pytest.skip(reason="probably bugged")
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
                "a": [1, 2, 3, 1, 2, 3, 4],
                "b": [1, -1, 3, 2, 5, 0, None],
                "i": [0, 1, 2, 3, 4, 5, 6],
                "g": [1, 1, 1, 1, 1, 1, 1],
            }
        )
    )
    result = df.with_columns(
        nw.col("a").cum_max(reverse=reverse).over("g", order_by="b")
    ).sort("i")
    expected = {
        "a": expected_a,
        "b": [1, -1, 3, 2, 5, 0, None],
        "i": [0, 1, 2, 3, 4, 5, 6],
        "g": [1, 1, 1, 1, 1, 1, 1],
    }
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("reverse", "expected_a"), [(False, [2, 2, 3]), (True, [3, 3, 3])]
)
def test_lazy_cum_max_ungrouped(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    *,
    reverse: bool,
    expected_a: list[int],
) -> None:
    if "dask" in str(constructor) and reverse:
        # https://github.com/dask/dask/issues/11802
        request.applymarker(pytest.mark.xfail)
    if "modin" in str(constructor):
        pytest.skip(reason="probably bugged")
    if ("polars" in str(constructor) and POLARS_VERSION < (1, 9)) or (
        "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3)
    ):
        pytest.skip(reason="too old version")

    df = nw.from_native(
        constructor({"a": [2, 3, 1], "b": [0, 2, 1], "i": [1, 2, 0]})
    ).sort("i")
    result = df.with_columns(
        nw.col("a").cum_max(reverse=reverse).over(order_by="b")
    ).sort("i")
    expected = {"a": expected_a, "b": [1, 0, 2], "i": [0, 1, 2]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("reverse", "expected_a"),
    [(False, [4, 4, 4, 4, 4, 4, 4]), (True, [3, 3, 3, 3, 2, 3, 4])],
)
def test_lazy_cum_max_ungrouped_ordered_by_nulls(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    *,
    reverse: bool,
    expected_a: list[int],
) -> None:
    if "dask" in str(constructor):
        # https://github.com/dask/dask/issues/11806
        request.applymarker(pytest.mark.xfail)
    if "modin" in str(constructor):
        pytest.skip(reason="probably bugged")
    if ("polars" in str(constructor) and POLARS_VERSION < (1, 9)) or (
        "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3)
    ):
        pytest.skip(reason="too old version")

    df = nw.from_native(
        constructor(
            {
                "a": [1, 2, 3, 1, 2, 3, 4],
                "b": [1, -1, 3, 2, 5, 0, None],
                "i": [0, 1, 2, 3, 4, 5, 6],
            }
        )
    ).sort("i")
    result = df.with_columns(
        nw.col("a").cum_max(reverse=reverse).over(order_by="b")
    ).sort("i")
    expected = {
        "a": expected_a,
        "b": [1, -1, 3, 2, 5, 0, None],
        "i": [0, 1, 2, 3, 4, 5, 6],
    }
    assert_equal_data(result, expected)


def test_cum_max_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if (PANDAS_VERSION < (2, 1)) and "pandas_pyarrow" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(
        cum_max=df["a"].cum_max(), reverse_cum_max=df["a"].cum_max(reverse=True)
    )
    assert_equal_data(result, expected)
