from __future__ import annotations

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
        any(x in str(constructor) for x in ("dask", "duckdb", "ibis"))
        and interpolation != "linear"
    ) or "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    q = 0.3
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw)

    context = (
        pytest.raises(
            NotImplementedError,
            match="`Expr.quantile` is not supported for Dask backend with multiple partitions.",
        )
        if "dask_lazy_p2" in str(constructor)
        else does_not_raise()
    )

    with context:
        result = df.select(nw.all().quantile(quantile=q, interpolation=interpolation))
        assert_equal_data(result, expected)


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
