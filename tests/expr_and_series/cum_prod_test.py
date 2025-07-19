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

data = {"a": [1, 2, None, 3]}

expected = {"cum_prod": [1, 2, None, 6], "reverse_cum_prod": [6, 6, None, 3]}


@pytest.mark.parametrize("reverse", [True, False])
def test_cum_prod_expr(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager, *, reverse: bool
) -> None:
    if (PANDAS_VERSION < (2, 1)) and "pandas_pyarrow" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    name = "reverse_cum_prod" if reverse else "cum_prod"
    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.col("a").cum_prod(reverse=reverse).alias(name))

    assert_equal_data(result, {name: expected[name]})


def test_cum_prod_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if (PANDAS_VERSION < (2, 1)) and "pandas_pyarrow" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(
        cum_prod=df["a"].cum_prod(), reverse_cum_prod=df["a"].cum_prod(reverse=True)
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("reverse", "expected_a"), [(False, [2, 2, 6]), (True, [3, 6, 3])]
)
def test_lazy_cum_prod_grouped(
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
    if "ibis" in str(constructor):
        # https://github.com/ibis-project/ibis/issues/10542
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
        nw.col("arg entina").cum_prod(reverse=reverse).over("g", order_by="ban gkok")
    ).sort("i ran")
    expected = {
        "arg entina": expected_a,
        "ban gkok": [1, 0, 2],
        "i ran": [0, 1, 2],
        "g": [1, 1, 1],
    }
    assert_equal_data(result, expected)
