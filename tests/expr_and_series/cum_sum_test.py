from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import POLARS_VERSION
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

data = {"a": [1, 2, None, 4]}
expected = {
    "cum_sum": [1, 3, None, 7],
    "reverse_cum_sum": [7, 6, None, 4],
}


@pytest.mark.parametrize("reverse", [True, False])
def test_cum_sum_expr(constructor_eager: ConstructorEager, *, reverse: bool) -> None:
    name = "reverse_cum_sum" if reverse else "cum_sum"
    df = nw.from_native(constructor_eager(data))
    result = df.select(
        nw.col("a").cum_sum(reverse=reverse).alias(name),
    )

    assert_equal_data(result, {name: expected[name]})


@pytest.mark.parametrize(
    ("reverse", "expected_a"),
    [
        (False, [3, 2, 6]),
        (True, [4, 6, 3]),
    ],
)
def test_lazy_cum_sum_grouped(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    *,
    reverse: bool,
    expected_a: list[int],
) -> None:
    if "duckdb" in str(constructor):
        # no window function support yet in duckdb
        request.applymarker(pytest.mark.xfail)
    if "pyarrow_table" in str(constructor):
        # grouped window functions not yet supported
        request.applymarker(pytest.mark.xfail)
    if "modin" in str(constructor):
        # bugged
        request.applymarker(pytest.mark.xfail)
    if "dask" in str(constructor):
        # https://github.com/dask/dask/issues/11806
        request.applymarker(pytest.mark.xfail)
    if "polars" in str(constructor) and POLARS_VERSION < (1, 9):
        pytest.skip(reason="too old version")

    df = nw.from_native(
        constructor(
            {
                "a": [1, 2, 3],
                "b": [1, 0, 2],
                "i": [0, 1, 2],
                "g": [1, 1, 1],
            }
        )
    )
    result = df.with_columns(
        nw.col("a").cum_sum(reverse=reverse).over("g", _order_by="b")
    ).sort("i")
    expected = {"a": expected_a, "b": [1, 0, 2], "i": [0, 1, 2]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("reverse", "expected_a"),
    [
        (False, [10, 6, 14, 11, 16, 9, 4]),
        (True, [7, 12, 5, 6, 2, 10, 16]),
    ],
)
def test_lazy_cum_sum_ordered_by_nulls(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    *,
    reverse: bool,
    expected_a: list[int],
) -> None:
    if "duckdb" in str(constructor):
        # no window function support yet in duckdb
        request.applymarker(pytest.mark.xfail)
    if "pyarrow_table" in str(constructor):
        # grouped window functions not yet supported
        request.applymarker(pytest.mark.xfail)
    if "modin" in str(constructor):
        # bugged
        request.applymarker(pytest.mark.xfail)
    if "dask" in str(constructor):
        # https://github.com/dask/dask/issues/11806
        request.applymarker(pytest.mark.xfail)
    if "polars" in str(constructor) and POLARS_VERSION < (1, 9):
        pytest.skip(reason="too old version")

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
        nw.col("a").cum_sum(reverse=reverse).over("g", _order_by="b")
    ).sort("i")
    expected = {
        "a": expected_a,
        "b": [1, -1, 3, 2, 5, 0, None],
        "i": [0, 1, 2, 3, 4, 5, 6],
    }
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("reverse", "expected_a"),
    [
        (False, [3, 2, 6]),
        (True, [4, 6, 3]),
    ],
)
def test_lazy_cum_sum_ungrouped(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    *,
    reverse: bool,
    expected_a: list[int],
) -> None:
    if "duckdb" in str(constructor):
        # no window function support yet in duckdb
        request.applymarker(pytest.mark.xfail)
    if "dask" in str(constructor) and reverse:
        # https://github.com/dask/dask/issues/11802
        request.applymarker(pytest.mark.xfail)
    if "modin" in str(constructor):
        # probably bugged
        request.applymarker(pytest.mark.xfail)
    if "polars" in str(constructor) and POLARS_VERSION < (1, 9):
        pytest.skip(reason="too old version")

    df = nw.from_native(
        constructor(
            {
                "a": [2, 3, 1],
                "b": [0, 2, 1],
                "i": [1, 2, 0],
            }
        )
    ).sort("i")
    result = df.with_columns(
        nw.col("a").cum_sum(reverse=reverse).over(_order_by="b")
    ).sort("i")
    expected = {"a": expected_a, "b": [1, 0, 2], "i": [0, 1, 2]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("reverse", "expected_a"),
    [
        (False, [10, 6, 14, 11, 16, 9, 4]),
        (True, [7, 12, 5, 6, 2, 10, 16]),
    ],
)
def test_lazy_cum_sum_ungrouped_ordered_by_nulls(
    constructor: Constructor,
    request: pytest.FixtureRequest,
    *,
    reverse: bool,
    expected_a: list[int],
) -> None:
    if "duckdb" in str(constructor):
        # no window function support yet in duckdb
        request.applymarker(pytest.mark.xfail)
    if "dask" in str(constructor):
        # https://github.com/dask/dask/issues/11806
        request.applymarker(pytest.mark.xfail)
    if "modin" in str(constructor):
        # probably bugged
        request.applymarker(pytest.mark.xfail)
    if "polars" in str(constructor) and POLARS_VERSION < (1, 9):
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
        nw.col("a").cum_sum(reverse=reverse).over(_order_by="b")
    ).sort("i")
    expected = {
        "a": expected_a,
        "b": [1, -1, 3, 2, 5, 0, None],
        "i": [0, 1, 2, 3, 4, 5, 6],
    }
    assert_equal_data(result, expected)


def test_cum_sum_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(
        cum_sum=df["a"].cum_sum(),
        reverse_cum_sum=df["a"].cum_sum(reverse=True),
    )
    assert_equal_data(result, expected)
