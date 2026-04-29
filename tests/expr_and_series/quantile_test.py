from __future__ import annotations

import re
from contextlib import nullcontext as does_not_raise
from typing import Literal

import pytest

import narwhals as nw
from tests.utils import Constructor, ConstructorEager, assert_equal_data


@pytest.mark.parametrize(
    ("interpolation", "expected"),
    [
        ("lower", {"a": [1.0], "b": [4.0], "z": [7.0]}),
        ("higher", {"a": [2.0], "b": [4.0], "z": [8.0]}),
        ("midpoint", {"a": [1.5], "b": [4.0], "z": [7.5]}),
        ("linear", {"a": [1.6], "b": [4.0], "z": [7.6]}),
        ("nearest", {"a": [2.0], "b": [4.0], "z": [8.0]}),
    ],
)
@pytest.mark.filterwarnings("ignore:the `interpolation=` argument to percentile")
def test_quantile_expr(
    constructor: Constructor,
    interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    expected: dict[str, list[float]],
    request: pytest.FixtureRequest,
) -> None:
    if (
        any(x in str(constructor) for x in ("dask", "duckdb", "ibis", "pyspark"))
        and interpolation != "linear"
    ):
        request.applymarker(pytest.mark.xfail)
    if "sqlframe" in str(constructor):
        # bug in sqlframes: https://github.com/eakmanrq/sqlframe/issues/625
        request.applymarker(pytest.mark.xfail)

    q = 0.3
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw)

    msg = re.escape(
        "`Expr.quantile` is not supported for Dask backend with multiple partitions."
    )
    context = (
        pytest.raises(NotImplementedError, match=msg)
        if "dask_lazy_p2" in str(constructor)
        else does_not_raise()
    )

    with context:
        result = df.select(nw.all().quantile(quantile=q, interpolation=interpolation))
        assert_equal_data(result, expected)


def test_quantile_expr_group_by(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(
        x in str(constructor) for x in ("dask", "duckdb", "ibis", "pyspark", "pyarrow")
    ):
        request.applymarker(pytest.mark.xfail)

    expected = {"a": [1, 2, 3], "b": [4.0, 6.0, 4.0]}
    q = 0.3
    data = {"a": [1, 2, 3], "b": [4, 6, 4]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw)

    result = df.group_by("a").agg(
        nw.col("b").quantile(quantile=q, interpolation="linear")
    )
    assert_equal_data(result.sort("a"), expected)


@pytest.mark.parametrize(
    ("interpolation", "expected"),
    [
        ("lower", 7.0),
        ("higher", 8.0),
        ("midpoint", 7.5),
        ("linear", 7.6),
        ("nearest", 8.0),
    ],
)
@pytest.mark.filterwarnings("ignore:the `interpolation=` argument to percentile")
def test_quantile_series(
    constructor_eager: ConstructorEager,
    interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    expected: float,
) -> None:
    q = 0.3

    series = nw.from_native(constructor_eager({"a": [7.0, 8.0, 9.0]}), eager_only=True)[
        "a"
    ].alias("a")
    result = series.quantile(quantile=q, interpolation=interpolation)
    assert_equal_data({"a": [result]}, {"a": [expected]})
