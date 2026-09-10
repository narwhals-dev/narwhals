from __future__ import annotations

import re

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError
from tests.utils import DUCKDB_VERSION, Constructor, ConstructorEager, assert_equal_data

# `a` has an even number of non-null values, where an approximate median disagrees
# with the exact one.
data = {
    "a": [3, 8, None, None],
    "b": [5, 5, None, 7],
    "z": [7.0, 8.0, 9.0, None],
    "s": ["f", "a", "x", "x"],
}
group_data = {"g": [1, 1, 2, 2], "a": [1.0, 2.0, 10.0, 20.0]}


@pytest.mark.parametrize(
    "expr", [nw.col("a", "b", "z").median(), nw.median("a", "b", "z")]
)
def test_median_expr(
    constructor: Constructor, expr: nw.Expr, request: pytest.FixtureRequest
) -> None:
    if "dask_lazy_p2" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = df.select(expr)
    expected = {"a": [5.5], "b": [5.0], "z": [8.0]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(("col", "expected"), [("a", 5.5), ("b", 5.0), ("z", 8.0)])
def test_median_series(
    constructor_eager: ConstructorEager, col: str, expected: float
) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)[col]
    result = series.median()
    assert_equal_data({col: [result]}, {col: [expected]})


def test_median_group_by(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "pyarrow_table" in str(constructor):
        reason = (
            "PyArrow has no exact hashed quantile kernel, only `hash_approximate_median` "
            " and `hash_tdigest`, so a grouped median stays approximate there. "
            "See: https://github.com/apache/arrow/issues/28985"
        )
        request.applymarker(pytest.mark.xfail(reason=reason))
    df = nw.from_native(constructor(group_data))
    result = df.group_by("g").agg(nw.col("a").median()).sort("g")
    assert_equal_data(result, {"g": [1, 2], "a": [1.5, 15.0]})


def test_median_over(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "duckdb" in str(constructor) and DUCKDB_VERSION < (1, 3):
        pytest.skip()
    if "pyarrow_table" in str(constructor):
        request.applymarker(pytest.mark.xfail(reason="pyarrow median over differs"))
    df = nw.from_native(constructor({"g": [1, 1, 2], "a": [1.0, 2.0, 3.0]}))
    result = df.with_columns(m=nw.col("a").median().over("g")).sort("g", "a")
    assert_equal_data(
        result, {"g": [1, 1, 2], "a": [1.0, 2.0, 3.0], "m": [1.5, 1.5, 3.0]}
    )


@pytest.mark.parametrize(
    "data", [{"a": [None, None]}, {"a": []}], ids=["all_null", "empty"]
)
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning:numpy")
def test_median_null_or_empty(
    constructor: Constructor, data: dict[str, list[int]]
) -> None:
    if "ibis" in str(constructor):
        pytest.skip(reason="ibis cannot create all-null column")
    df = nw.from_native(constructor(data))
    result = df.with_columns(nw.col("a").cast(nw.Float64())).select(nw.col("a").median())
    assert_equal_data(result, {"a": [None]})


def test_median_boolean(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    # Only Spark still rejects non-numeric input; dask cannot pin the exact
    # approximate value across partitions.
    if "pyspark" in str(constructor) and "sqlframe" not in str(constructor):
        request.applymarker(pytest.mark.xfail(reason="boolean median"))
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail(reason="approximate median"))
    df = nw.from_native(constructor({"b": [True, False, True]}))
    result = df.select(nw.col("b").median())
    assert_equal_data(result, {"b": [1.0]})


@pytest.mark.parametrize("expr", [nw.col("s").median(), nw.median("s")])
def test_median_expr_raises_on_str(
    constructor: Constructor, expr: nw.Expr, request: pytest.FixtureRequest
) -> None:
    if (
        ("pyspark" in str(constructor))
        or "duckdb" in str(constructor)
        or "ibis" in str(constructor)
    ):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    if isinstance(df, nw.LazyFrame):
        with pytest.raises(
            InvalidOperationError, match="`median` operation not supported"
        ):
            df.select(expr).lazy().collect()
    else:
        with pytest.raises(
            InvalidOperationError, match="`median` operation not supported"
        ):
            df.select(expr)


@pytest.mark.parametrize(("col"), [("s")])
def test_median_series_raises_on_str(
    constructor_eager: ConstructorEager, col: str
) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)[col]
    with pytest.raises(
        InvalidOperationError,
        match=re.escape("`median` operation not supported for non-numeric input type."),
    ):
        series.median()
