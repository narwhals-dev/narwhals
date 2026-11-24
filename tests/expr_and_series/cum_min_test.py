from __future__ import annotations

import math

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from tests.utils import (
    DUCKDB_VERSION,
    PANDAS_VERSION,
    POLARS_VERSION,
    PYARROW_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
    is_windows,
    uses_pyarrow_backend,
)

data = {"a": [3, 1, None, 2]}

expected = {"cum_min": [3, 1, None, 1], "reverse_cum_min": [1, 1, None, 2]}


@pytest.mark.parametrize("reverse", [True, False])
def test_cum_min_expr(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager, *, reverse: bool
) -> None:
    if (PANDAS_VERSION < PkgVersion("2.1")) and "pandas_pyarrow" in str(
        constructor_eager
    ):
        request.applymarker(pytest.mark.xfail)

    name = "reverse_cum_min" if reverse else "cum_min"
    df = nw.from_native(constructor_eager(data))
    result = df.select(nw.col("a").cum_min(reverse=reverse).alias(name))

    assert_equal_data(result, {name: expected[name]})


@pytest.mark.parametrize(
    ("reverse", "expected_a"), [(False, [1, 2, 1]), (True, [1, 1, 3])]
)
def test_lazy_cum_min_grouped(
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
    if ("polars" in str(constructor) and POLARS_VERSION < PkgVersion("1.9")) or (
        "duckdb" in str(constructor) and DUCKDB_VERSION < PkgVersion("1.3")
    ):
        pytest.skip(reason="too old version")
    if "cudf" in str(constructor):
        # https://github.com/rapidsai/cudf/issues/18159
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(
        constructor({"a": [1, 2, 3], "b": [1, 0, 2], "i": [0, 1, 2], "g": [1, 1, 1]})
    )
    result = df.with_columns(
        nw.col("a").cum_min(reverse=reverse).over("g", order_by="b")
    ).sort("i")
    expected = {"a": expected_a, "b": [1, 0, 2], "i": [0, 1, 2], "g": [1, 1, 1]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("reverse", "expected_a"),
    [(False, [1, 2, 1, None, 1, 2, 4]), (True, [1, 1, 2, None, 2, 1, 1])],
)
def test_lazy_cum_min_ordered_by_nulls(
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
    if "pandas_nullable" in str(constructor):
        # https://github.com/pandas-dev/pandas/issues/62473
        request.applymarker(pytest.mark.xfail)
    if "dask" in str(constructor):
        # https://github.com/dask/dask/issues/11806
        request.applymarker(pytest.mark.xfail)
    if ("polars" in str(constructor) and POLARS_VERSION < PkgVersion("1.9")) or (
        "duckdb" in str(constructor) and DUCKDB_VERSION < PkgVersion("1.3")
    ):
        pytest.skip(reason="too old version")
    if "cudf" in str(constructor):
        # https://github.com/rapidsai/cudf/issues/18159
        request.applymarker(pytest.mark.xfail)
    if (
        "pyarrow" in str(constructor)
        and is_windows()
        and PYARROW_VERSION < PkgVersion("22")
    ):
        # https://github.com/pandas-dev/pandas/issues/62477
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(
        constructor(
            {
                "a": [1, 2, 3, None, 2, 3, 4],
                "b": [1, -1, 3, 2, 5, 0, None],
                "i": [0, 1, 2, 3, 4, 5, 6],
                "g": [1, 1, 1, 1, 1, 1, 1],
            }
        )
    )
    result = df.with_columns(
        nw.col("a").cum_min(reverse=reverse).over("g", order_by="b")
    ).sort("i")
    expected = {
        "a": expected_a,
        "b": [1, -1, 3, 2, 5, 0, None],
        "i": [0, 1, 2, 3, 4, 5, 6],
        "g": [1, 1, 1, 1, 1, 1, 1],
    }
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("reverse", "expected_a"), [(False, [1, 2, 1]), (True, [1, 1, 3])]
)
def test_lazy_cum_min_ungrouped(
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
    if ("polars" in str(constructor) and POLARS_VERSION < PkgVersion("1.9")) or (
        "duckdb" in str(constructor) and DUCKDB_VERSION < PkgVersion("1.3")
    ):
        pytest.skip(reason="too old version")
    df = nw.from_native(
        constructor({"a": [2, 3, 1], "b": [0, 2, 1], "i": [1, 2, 0]})
    ).sort("i")
    result = df.with_columns(
        nw.col("a").cum_min(reverse=reverse).over(order_by="b")
    ).sort("i")
    expected = {"a": expected_a, "b": [1, 0, 2], "i": [0, 1, 2]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("reverse", "expected_a"),
    [(False, [1, 2, 1, 1, 1, 2, 4]), (True, [1, 1, 2, 1, 2, 1, 1])],
)
def test_lazy_cum_min_ungrouped_ordered_by_nulls(
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
    if ("polars" in str(constructor) and POLARS_VERSION < PkgVersion("1.9")) or (
        "duckdb" in str(constructor) and DUCKDB_VERSION < PkgVersion("1.3")
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
        nw.col("a").cum_min(reverse=reverse).over(order_by="b")
    ).sort("i")
    expected = {
        "a": expected_a,
        "b": [1, -1, 3, 2, 5, 0, None],
        "i": [0, 1, 2, 3, 4, 5, 6],
    }
    assert_equal_data(result, expected)


def test_cum_min_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if (PANDAS_VERSION < PkgVersion("2.1")) and "pandas_pyarrow" in str(
        constructor_eager
    ):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.select(
        cum_min=df["a"].cum_min(), reverse_cum_min=df["a"].cum_min(reverse=True)
    )
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("values", "expected_cum_min", "expected_reverse_cum_min"),
    [
        ([float("inf")], [float("inf")], [float("inf")]),
        ([float("inf"), 3.0, 4.0], [float("inf"), 3.0, 3.0], [3.0, 3.0, 4.0]),
        ([float("inf"), None, 3.0], [float("inf"), None, 3.0], [3.0, None, 3.0]),
    ],
)
@pytest.mark.parametrize("reverse", [True, False])
def test_cum_min_infinite_floats(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    values: list[float | None],
    expected_cum_min: list[float | None],
    expected_reverse_cum_min: list[float | None],
    *,
    reverse: bool,
) -> None:
    if uses_pyarrow_backend(constructor_eager) and PYARROW_VERSION < PkgVersion("26"):
        # pandas and Modin call the PyArrow kernel directly: https://github.com/apache/arrow/issues/51194
        pytest.skip(reason="affected by PyArrow bug fixed upstream in 26.0")
    if "polars" in str(constructor_eager) and POLARS_VERSION < PkgVersion("1.31"):
        # Polars<1.31 seeds the cumulative kernel with the largest finite value, so a
        # leading (in accumulation order) infinity is replaced by it.
        ordered = values[::-1] if reverse else values
        if math.isinf(next(v for v in ordered if v is not None)):
            request.applymarker(
                pytest.mark.xfail(reason="fixed upstream in Polars 1.31", strict=True)
            )

    df = nw.from_native(constructor_eager({"a": values}), eager_only=True)
    result = df.select(nw.col("a").cum_min(reverse=reverse))

    expected = expected_reverse_cum_min if reverse else expected_cum_min
    assert_equal_data(result, {"a": expected})
