from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import Literal

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

rank_methods = ["average", "min", "max", "dense", "ordinal"]

data_int = {"a": [3, 6, 1, 1, None, 6], "b": [1, 1, 2, 1, 2, 2], "i": [1, 2, 3, 4, 5, 6]}
data_float = {
    "a": [3.1, 6.1, 1.5, 1.5, None, 6.1],
    "b": [1, 1, 2, 1, 2, 2],
    "i": [1, 2, 3, 4, 5, 6],
}

expected = {
    "average": [3.0, 4.5, 1.5, 1.5, None, 4.5],
    "min": [3, 4, 1, 1, None, 4],
    "max": [3, 5, 2, 2, None, 5],
    "dense": [2, 3, 1, 1, None, 3],
    "ordinal": [3, 4, 1, 2, None, 5],
}

expected_desc = {
    "average": [3.0, 1.5, 4.5, 4.5, None, 1.5],
    "min": [3, 1, 4, 4, None, 1],
    "max": [3, 2, 5, 5, None, 2],
    "dense": [2, 1, 3, 3, None, 1],
    "ordinal": [3, 1, 4, 5, None, 2],
}

expected_over = {
    "average": [2.0, 3.0, 1.0, 1.0, None, 2.0],
    "min": [2, 3, 1, 1, None, 2],
    "max": [2, 3, 1, 1, None, 2],
    "dense": [2, 3, 1, 1, None, 2],
    "ordinal": [2, 3, 1, 1, None, 2],
}

expected_over_desc = {
    "average": [2.0, 1.0, 2.0, 3.0, None, 1.0],
    "min": [2, 1, 2, 3, None, 1],
    "max": [2, 1, 2, 3, None, 1],
    "dense": [2, 1, 2, 3, None, 1],
    "ordinal": [2, 1, 2, 3, None, 1],
}


@pytest.mark.parametrize("method", rank_methods)
@pytest.mark.parametrize("data", [data_int, data_float])
def test_rank_expr(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    method: Literal["average", "min", "max", "dense", "ordinal"],
    data: dict[str, list[float]],
) -> None:
    if (
        "pandas_pyarrow" in str(constructor_eager)
        and PANDAS_VERSION < (2, 1)
        and isinstance(data["a"][0], int)
    ):
        request.applymarker(pytest.mark.xfail)

    context = (
        pytest.raises(
            ValueError,
            match=r"`rank` with `method='average' is not supported for pyarrow backend.",
        )
        if "pyarrow_table" in str(constructor_eager) and method == "average"
        else does_not_raise()
    )

    with context:
        df = nw.from_native(constructor_eager(data))

        result = df.select(nw.col("a").rank(method=method))
        expected_data = {"a": expected[method]}
        assert_equal_data(result, expected_data)


@pytest.mark.parametrize("method", rank_methods)
@pytest.mark.parametrize("data", [data_int, data_float])
def test_rank_series(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    method: Literal["average", "min", "max", "dense", "ordinal"],
    data: dict[str, list[float]],
) -> None:
    if (
        "pandas_pyarrow" in str(constructor_eager)
        and PANDAS_VERSION < (2, 1)
        and isinstance(data["a"][0], int)
    ):
        request.applymarker(pytest.mark.xfail)

    context = (
        pytest.raises(
            ValueError,
            match=r"`rank` with `method='average' is not supported for pyarrow backend.",
        )
        if "pyarrow_table" in str(constructor_eager) and method == "average"
        else does_not_raise()
    )

    with context:
        df = nw.from_native(constructor_eager(data), eager_only=True)

        result = {"a": df["a"].rank(method=method)}
        expected_data = {"a": expected[method]}
        assert_equal_data(result, expected_data)


@pytest.mark.parametrize("method", rank_methods)
def test_rank_expr_in_over_context(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    method: Literal["average", "min", "max", "dense", "ordinal"],
) -> None:
    if any(x in str(constructor) for x in ("pyarrow_table", "dask", "cudf")):
        # Pyarrow raises:
        # > pyarrow.lib.ArrowKeyError: No function registered with name: hash_rank
        # We can handle that to provide a better error message.
        # cudf: https://github.com/rapidsai/cudf/issues/18159
        request.applymarker(pytest.mark.xfail)

    if "pandas_pyarrow" in str(constructor) and PANDAS_VERSION < (2, 1):
        pytest.skip(reason="bug in old version")
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip(reason="too old version")

    df = nw.from_native(constructor(data_float))

    result = (
        df.with_columns(a=nw.col("a").rank(method=method).over("b")).sort("i").select("a")
    )
    expected_data = {"a": expected_over[method]}
    assert_equal_data(result, expected_data)


def test_invalid_method_raise(constructor_eager: ConstructorEager) -> None:
    method = "invalid_method_name"
    df = nw.from_native(constructor_eager(data_float))

    msg = (
        "Ranking method must be one of {'average', 'min', 'max', 'dense', 'ordinal'}. "
        f"Found '{method}'"
    )

    with pytest.raises(ValueError, match=msg):
        df.select(nw.col("a").rank(method=method))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=msg):
        df.lazy().collect()["a"].rank(method=method)  # type: ignore[arg-type]


@pytest.mark.parametrize("method", rank_methods)
@pytest.mark.parametrize("data", [data_int, data_float])
def test_lazy_rank_expr(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    method: Literal["average", "min", "max", "dense", "ordinal"],
    data: dict[str, list[float]],
) -> None:
    if (
        "pandas_pyarrow" in str(constructor)
        and PANDAS_VERSION < (2, 1)
        and isinstance(data["a"][0], int)
    ):
        request.applymarker(pytest.mark.xfail)

    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        request.applymarker(pytest.mark.xfail)

    if "dask" in str(constructor):
        # `rank` is not implemented in Dask
        request.applymarker(pytest.mark.xfail)

    context = (
        pytest.raises(
            ValueError,
            match=r"`rank` with `method='average' is not supported for pyarrow backend.",
        )
        if "pyarrow_table" in str(constructor) and method == "average"
        else does_not_raise()
    )

    with context:
        df = nw.from_native(constructor(data))

        result = df.with_columns(a=nw.col("a").rank(method=method)).sort("i").select("a")
        expected_data = {"a": expected[method]}
        assert_equal_data(result, expected_data)


@pytest.mark.parametrize("method", rank_methods)
@pytest.mark.parametrize("data", [data_int, data_float])
def test_lazy_rank_expr_desc(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    method: Literal["average", "min", "max", "dense", "ordinal"],
    data: dict[str, list[float]],
) -> None:
    if (
        "pandas_pyarrow" in str(constructor)
        and PANDAS_VERSION < (2, 1)
        and isinstance(data["a"][0], int)
    ):
        request.applymarker(pytest.mark.xfail)

    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        request.applymarker(pytest.mark.xfail)

    if "dask" in str(constructor):
        # `rank` is not implemented in Dask
        request.applymarker(pytest.mark.xfail)

    context = (
        pytest.raises(
            ValueError,
            match=r"`rank` with `method='average' is not supported for pyarrow backend.",
        )
        if "pyarrow_table" in str(constructor) and method == "average"
        else does_not_raise()
    )

    with context:
        df = nw.from_native(constructor(data))

        result = (
            df.with_columns(a=nw.col("a").rank(method=method, descending=True))
            .sort("i")
            .select("a")
        )
        expected_data = {"a": expected_desc[method]}
        assert_equal_data(result, expected_data)


@pytest.mark.parametrize("method", rank_methods)
def test_rank_expr_in_over_desc(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    method: Literal["average", "min", "max", "dense", "ordinal"],
) -> None:
    if any(x in str(constructor) for x in ("pyarrow_table", "dask", "cudf")):
        # Pyarrow raises:
        # > pyarrow.lib.ArrowKeyError: No function registered with name: hash_rank
        # We can handle that to provide a better error message.
        # cudf: https://github.com/rapidsai/cudf/issues/18159
        request.applymarker(pytest.mark.xfail)

    if "pandas_pyarrow" in str(constructor) and PANDAS_VERSION < (2, 1):
        pytest.skip(reason="bug in old version")
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip(reason="too old version")

    df = nw.from_native(constructor(data_float))

    result = (
        df.with_columns(a=nw.col("a").rank(method=method, descending=True).over("b"))
        .sort("i")
        .select("a")
    )
    expected_data = {"a": expected_over_desc[method]}
    assert_equal_data(result, expected_data)


def test_rank_with_order_by(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "dask" in str(constructor):
        # `rank` is not implemented in Dask
        request.applymarker(pytest.mark.xfail)
    if "pandas_pyarrow" in str(constructor) and PANDAS_VERSION < (2, 1):
        pytest.skip(reason="bug in old version")
    if "polars" in str(constructor) and POLARS_VERSION < (1, 10):
        pytest.skip(reason="too old")
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip(reason="too old version")

    df = nw.from_native(
        constructor(
            {"a": [1, 1, 2, 2, 3, 3], "b": [3, None, 4, 3, 5, 6], "i": list(range(6))}
        )
    )
    result = df.with_columns(c=nw.col("a").rank("ordinal").over(order_by="b")).sort("i")
    expected = {
        "a": [1, 1, 2, 2, 3, 3],
        "b": [3, None, 4, 3, 5, 6],
        "i": [0, 1, 2, 3, 4, 5],
        "c": [2, 1, 4, 3, 5, 6],
    }
    assert_equal_data(result, expected)


def test_rank_with_order_by_and_partition_by(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(x in str(constructor) for x in ("dask", "pyarrow_table", "cudf")):
        # `rank` is not implemented in Dask
        # pyarrow only supports aggregations in `over(partition_by=...)`
        # cudf: https://github.com/rapidsai/cudf/issues/18159
        request.applymarker(pytest.mark.xfail)
    if "pandas_pyarrow" in str(constructor) and PANDAS_VERSION < (2, 1):
        pytest.skip(reason="bug in old version")
    if "polars" in str(constructor) and POLARS_VERSION < (1, 10):
        pytest.skip(reason="too old")
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip(reason="too old version")

    df = nw.from_native(
        constructor(
            {
                "a": [1, 1, 2, 2, 3, 3],
                "b": [3, None, 4, 3, 5, 6],
                "i": list(range(6)),
                "g": ["x", "x", "x", "y", "y", "y"],
            }
        )
    )
    result = df.with_columns(c=nw.col("a").rank("ordinal").over("g", order_by="b")).sort(
        "i"
    )
    expected = {
        "a": [1, 1, 2, 2, 3, 3],
        "b": [3, None, 4, 3, 5, 6],
        "i": [0, 1, 2, 3, 4, 5],
        "g": ["x", "x", "x", "y", "y", "y"],
        "c": [2, 1, 3, 1, 2, 3],
    }
    assert_equal_data(result, expected)
