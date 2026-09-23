from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import (
    POLARS_VERSION,
    PYARROW_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
)

# Ties for 0, 1 and 2 decimals, all exactly representable as floats so that Python's
# `round` (which rounds half to even) is a valid reference for every backend.
TIES = [0.5, 1.5, 2.5, -0.5, -2.5, 0.25, 0.75, -0.75, 0.125, 0.375, -0.375]


@pytest.mark.parametrize("decimals", [0, 1, 2])
def test_round(constructor: Constructor, decimals: int) -> None:
    data = {"a": [*TIES, 2.12345, 2.56789, 3.901234]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw)

    expected_data = {k: [round(e, decimals) for e in v] for k, v in data.items()}
    result_frame = df.select(nw.col("a").round(decimals))
    assert_equal_data(result_frame, expected_data)


@pytest.mark.parametrize("decimals", [0, 1, 2])
def test_round_series(constructor_eager: ConstructorEager, decimals: int) -> None:
    data = {"a": [*TIES, 1.12345, 2.56789, 3.901234]}
    df_raw = constructor_eager(data)
    df = nw.from_native(df_raw, eager_only=True)

    expected_data = {k: [round(e, decimals) for e in v] for k, v in data.items()}
    result_series = df["a"].round(decimals)

    assert_equal_data({"a": result_series}, expected_data)


def test_round_special_values(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "ibis" in str(constructor):
        reason = "Ibis' `round` with zero digits returns `Int64`, which can't hold NaN."
        request.applymarker(pytest.mark.xfail(reason=reason))
    nan, inf = float("nan"), float("inf")
    data = {"a": [0.5, None, nan, inf, -inf]}
    result = nw.from_native(constructor(data)).select(nw.col("a").round(0))
    assert_equal_data(result, {"a": [0.0, None, nan, inf, -inf]})


def test_round_large_decimals(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if any(s in str(constructor) for s in ("pandas", "modin", "pyarrow", "dask")):
        reason = "NumPy and Arrow overflow on a scale this large, independently of tie correction."
        request.applymarker(pytest.mark.xfail(reason=reason))
    if POLARS_VERSION < (1, 27) and "polars" in str(constructor):
        reason = "Polars' own `round` overflowed to NaN past 308 decimals before 1.27."
        request.applymarker(pytest.mark.xfail(reason=reason))
    # `10.0**decimals` overflows past 308, which used to raise before the scale was
    # clamped. Rounding a float that far out is the identity.
    data = {"a": [1.5, 2.5]}
    result = nw.from_native(constructor(data)).select(nw.col("a").round(400))
    assert_equal_data(result, data)


@pytest.mark.parametrize("decimals", [0, 1])
def test_round_keeps_integers(
    constructor: Constructor, decimals: int, request: pytest.FixtureRequest
) -> None:
    corrects_ties = (
        any(s in str(constructor) for s in ("duckdb", "sqlframe", "pyspark"))
        or ("ibis" in str(constructor) and decimals != 0)
        or (POLARS_VERSION < (1, 29) and "polars" in str(constructor))
    )
    if corrects_ties:
        reason = (
            "Correcting a native `round` that breaks ties away from zero takes float "
            "arithmetic, which widens the result and loses integers above 2**53."
        )
        request.applymarker(pytest.mark.xfail(reason=reason))
    if PYARROW_VERSION < (14,) and "pyarrow" in str(constructor):
        reason = "`pc.round` returned a float for integer input, and raised above 2**53."
        request.applymarker(pytest.mark.xfail(reason=reason))
    data = {"a": [1, -2, 9007199254740993]}
    result = nw.from_native(constructor(data)).select(nw.col("a").round(decimals))
    assert result.lazy().collect_schema()["a"] == nw.Int64
    assert_equal_data(result, data)
