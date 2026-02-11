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

data = {"a": ["x", "y", None, "z"]}

expected = {"cum_count": [1, 2, 2, 3], "reverse_cum_count": [3, 2, 1, 1]}


@pytest.mark.parametrize("reverse", [True, False])
def test_cum_count_expr(constructor_eager: ConstructorEager, *, reverse: bool) -> None:
    name = "reverse_cum_count" if reverse else "cum_count"
    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.col("a").cum_count(reverse=reverse).alias(name))

    assert_equal_data(result, {name: expected[name]})


def test_cum_count_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(
        cum_count=df["a"].cum_count(), reverse_cum_count=df["a"].cum_count(reverse=True)
    )
    expected = {"cum_count": [1, 2, 2, 3], "reverse_cum_count": [3, 2, 1, 1]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("reverse", "expected_a"), [(False, [1, 1, 2]), (True, [1, 2, 1])]
)
def test_lazy_cum_count_grouped(
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
                "arg entina": [None, 2, 3],
                "ban gkok": [1, 0, 2],
                "i ran": [0, 1, 2],
                "g": [1, 1, 1],
            }
        )
    )
    result = df.with_columns(
        nw.col("arg entina").cum_count(reverse=reverse).over("g", order_by="ban gkok")
    ).sort("i ran")
    expected = {
        "arg entina": expected_a,
        "ban gkok": [1, 0, 2],
        "i ran": [0, 1, 2],
        "g": [1, 1, 1],
    }
    assert_equal_data(result, expected)
