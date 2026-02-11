from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from tests.utils import DUCKDB_VERSION, PYARROW_VERSION, assert_equal_data

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager

data = {
    "a": [1, 1, 1, 2, 2, 3],
    "b": [1, 2, 3, 4, 5, 6],
    "c": [None, None, 1, None, 2, None],
}


@pytest.mark.parametrize("ignore_nulls", [False, True])
def test_any_value_expr(
    constructor: Constructor, request: pytest.FixtureRequest, *, ignore_nulls: bool
) -> None:
    if "dask" in str(constructor):
        reason = "sample does not allow n, use frac instead"
        request.applymarker(pytest.mark.xfail(reason=reason))

    df = nw.from_native(constructor(data))

    # Aggregation
    result = (
        df.select(nw.col("a", "b").any_value(ignore_nulls=ignore_nulls)).lazy().collect()
    )
    assert result.shape == (1, 2)

    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        reason = "broadcast requires `over`, which requires DuckDB 1.3.0"
        pytest.skip(reason=reason)

    # Aggregation + right broadcast
    result = (
        df.select(nw.col("a"), nw.col("b").any_value(ignore_nulls=ignore_nulls))
        .lazy()
        .collect()
    )
    assert result.shape == (6, 2)
    assert result["b"].n_unique() == 1

    # Aggregation + left broadcast
    result = (
        df.select(nw.col("a").any_value(ignore_nulls=ignore_nulls), nw.col("b"))
        .lazy()
        .collect()
    )
    assert result.shape == (6, 2)
    assert result["a"].n_unique() == 1

    # Test with column containing nulls - ignore_nulls=True should return non-null value
    if ignore_nulls:
        result = df.select(nw.col("c").any_value(ignore_nulls=True)).lazy().collect()
        assert result.shape == (1, 1)

        value = result["c"][0]
        assert value in {1, 2}


@pytest.mark.parametrize("ignore_nulls", [False, True])
def test_any_value_series(
    constructor_eager: ConstructorEager, *, ignore_nulls: bool
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)

    result = {
        "a": [df["a"].any_value(ignore_nulls=ignore_nulls)],
        "b": [df["b"].any_value(ignore_nulls=ignore_nulls)],
    }
    assert all(len(r) == 1 for r in result.values())

    # Test with column containing nulls - ignore_nulls=True should return non-null value
    if ignore_nulls:
        result_c = df["c"].any_value(ignore_nulls=True)
        assert result_c in {1, 2}


@pytest.mark.parametrize("ignore_nulls", [False, True])
def test_any_value_group_by(
    constructor: Constructor, request: pytest.FixtureRequest, *, ignore_nulls: bool
) -> None:
    if "dask" in str(constructor):
        reason = "sample does not allow n, use frac instead"
        request.applymarker(pytest.mark.xfail(reason=reason))
    if "pyarrow_table" in str(constructor) and PYARROW_VERSION < (14, 0):
        reason = "too old"
        pytest.skip(reason)

    df = nw.from_native(constructor(data))

    if ignore_nulls and df.implementation.is_pandas_like():
        reason = "not implemented"
        request.applymarker(pytest.mark.xfail(reason=reason))

    result = df.group_by("a").agg(
        nw.col("b").any_value(ignore_nulls=ignore_nulls).alias("any"), nw.col("b").max()
    )
    assert result.lazy().collect().shape == (3, 3)

    # Test with column containing nulls - ignore_nulls=True should return non-null value
    result = df.group_by("a").agg(nw.col("c").any_value(ignore_nulls=ignore_nulls))
    result_collected = result.lazy().collect().sort("a")
    assert result_collected.shape == (3, 2)

    if ignore_nulls:
        values = result_collected["c"].to_list()
        assert values[0] == 1  # group a=1: [None, None, 1] -> 1
        assert values[1] == 2  # group a=2: [None, 2] -> 2
        assert values[2] is None  # group a=3: [None] -> None regardless of ignore_nulls


@pytest.mark.parametrize("ignore_nulls", [False, True])
def test_any_value_over(
    constructor: Constructor, request: pytest.FixtureRequest, *, ignore_nulls: bool
) -> None:
    if "dask" in str(constructor):
        reason = "sample does not allow n, use frac instead"
        request.applymarker(pytest.mark.xfail(reason=reason))

    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        reason = "`over` requires DuckDB 1.3.0"
        pytest.skip(reason=reason)

    if "pyarrow_table" in str(constructor) and PYARROW_VERSION < (14, 0):
        reason = "too old"
        pytest.skip(reason)

    df = nw.from_native(constructor(data))

    if ignore_nulls and df.implementation.is_pandas_like():
        reason = "not implemented"
        request.applymarker(pytest.mark.xfail(reason=reason))

    result = df.select(
        nw.col("a"), nw.col("b").any_value(ignore_nulls=ignore_nulls).over("a")
    )
    assert result.lazy().collect().shape == (6, 2)
    uniques = result.group_by("a").agg(nw.col("b").n_unique()).sort("a")
    assert_equal_data(uniques, {"a": [1, 2, 3], "b": [1, 1, 1]})

    # Test with column containing nulls
    if ignore_nulls:
        result = df.select(
            nw.col("a"), nw.col("c").any_value(ignore_nulls=True).over("a")
        )
        result_collected = result.lazy().collect().sort("a")
        assert result_collected.shape == (6, 2)
        # Group 1 should have value 1, group 2 should have value 2
        group_1_values = result_collected.filter(nw.col("a") == 1)["c"].to_list()
        group_2_values = result_collected.filter(nw.col("a") == 2)["c"].to_list()
        assert all(v == 1 for v in group_1_values)
        assert all(v == 2 for v in group_2_values)
