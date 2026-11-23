from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from tests.utils import POLARS_VERSION

if TYPE_CHECKING:
    from narwhals.dtypes import DType
    from tests.utils import Constructor, ConstructorEager

NUMERIC_REDUCTIONS = [
    "mean",
    "min",
    "max",
    "std",
    "var",
    "median",
    "sum",
    "skew",
    "kurtosis",
]


# pandas 2.x `median` over all-null floats warns via `np.nanmedian`.
@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.parametrize("reduction", NUMERIC_REDUCTIONS)
def test_numeric_reduction_keeps_dtype_when_all_null(
    constructor: Constructor, request: pytest.FixtureRequest, reduction: str
) -> None:
    if reduction != "sum" and any(
        x in str(constructor)
        for x in ("pandas_nullable", "pandas_pyarrow", "modin_pyarrow")
    ):
        reason = "Return an untyped `pd.NA`. `sum` returns a typed zero."
        request.applymarker(pytest.mark.xfail(reason=reason))
    if reduction in {"skew", "kurtosis"} and any(
        x in str(constructor) for x in ("pandas_constructor", "modin_constructor", "ibis")
    ):
        reason = "pandas-like returns an untyped `None`. Ibis does not implement them."
        request.applymarker(pytest.mark.xfail(reason=reason))

    df = nw.from_native(constructor({"a": [1.0, 2.0]}))
    all_null = df.with_columns(a=nw.when(nw.col("a") < 0).then(nw.col("a")))
    expr = getattr(nw.col("a"), reduction)()
    assert all_null.select(expr).collect_schema() == df.select(expr).collect_schema()


def test_string_reduction_keeps_dtype_when_all_null(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "pandas_constructor" in str(constructor):
        reason = "A reduction over an all-null object column returns a float."
        request.applymarker(pytest.mark.xfail(reason=reason))

    df = nw.from_native(constructor({"a": ["x", "y"]}))
    all_null = df.with_columns(a=nw.when(nw.col("a") == "zzz").then(nw.col("a")))
    assert all_null.select(nw.col("a").max()).collect_schema() == {"a": nw.String()}


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [
        ("min", int),
        ("max", int),
        ("sum", int),
        ("first", int),
        ("last", int),
        ("any_value", int),
        ("count", int),
        ("null_count", int),
        ("n_unique", int),
        ("len", int),
        ("arg_min", int),
        ("arg_max", int),
        ("mean", float),
        ("median", float),
        ("std", float),
        ("var", float),
        ("skew", float),
        ("kurtosis", float),
        ("any", bool),
        ("all", bool),
    ],
)
def test_series_aggregation_returns_python_literal(
    constructor_eager: ConstructorEager, aggregation: str, expected: type
) -> None:
    if any(x in str(constructor_eager) for x in ("pandas", "modin", "cudf")):
        pytest.skip(reason="pandas-like returns numpy scalars, which vary by version")

    df = nw.from_native(
        constructor_eager({"a": [1, 3, 2], "b": [True, False, True]}), eager_only=True
    )
    series = df["b"] if expected is bool else df["a"]
    assert type(getattr(series, aggregation)()) is expected


@pytest.mark.parametrize(
    "expr",
    [
        nw.col("a").min(),
        nw.col("a").max(),
        nw.col("a").first(),
        nw.col("a").last(),
        nw.col("a").any_value(),
        nw.col("a").first().over(order_by="i"),
    ],
    ids=["min", "max", "first", "last", "any_value", "first_over_order_by"],
)
@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        ([1, 3, 2], nw.Int32()),
        (
            [datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1)],
            nw.Datetime("ns"),
        ),
    ],
    ids=["Int32", "Datetime-ns"],
)
@pytest.mark.parametrize("empty", [False, True], ids=["values", "empty"])
def test_expr_aggregation_keeps_dtype(
    constructor_eager: ConstructorEager,
    expr: nw.Expr,
    values: list[Any],
    dtype: DType,
    *,
    empty: bool,
) -> None:
    if "polars" in str(constructor_eager) and POLARS_VERSION < PkgVersion("1.10"):
        pytest.skip(reason="`over(order_by=...)` requires Polars 1.10")
    name = str(constructor_eager)
    if any(x in name for x in ("pandas", "modin", "cudf")) and (
        empty or "pyarrow" in name
    ):
        pytest.skip(reason="pandas-like returns NaN, `pd.NA` or Python scalars here")

    df = nw.from_native(constructor_eager({"a": values, "i": [0, 1, 2]}), eager_only=True)
    df = df.with_columns(nw.col("a").cast(dtype))
    if empty:
        df = df.head(0)
    assert df.select(expr).schema == {"a": dtype}
