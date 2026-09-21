from __future__ import annotations

from decimal import Decimal

import pytest

import narwhals as nw
from tests.utils import DUCKDB_VERSION, Constructor, ConstructorEager, assert_equal_data


@pytest.mark.parametrize("decimals", [0, 1, 2])
def test_round(constructor: Constructor, decimals: int) -> None:
    data = {"a": [2.12345, 2.56789, 3.901234]}
    df_raw = constructor(data)
    df = nw.from_native(df_raw)

    expected_data = {k: [round(e, decimals) for e in v] for k, v in data.items()}
    result_frame = df.select(nw.col("a").round(decimals))
    assert_equal_data(result_frame, expected_data)


@pytest.mark.parametrize("decimals", [0, 1, 2])
def test_round_series(constructor_eager: ConstructorEager, decimals: int) -> None:
    data = {"a": [1.12345, 2.56789, 3.901234]}
    df_raw = constructor_eager(data)
    df = nw.from_native(df_raw, eager_only=True)

    expected_data = {k: [round(e, decimals) for e in v] for k, v in data.items()}
    result_series = df["a"].round(decimals)

    assert_equal_data({"a": result_series}, expected_data)


HALF_TIES = [
    (
        0,
        [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 0.4, 0.6],
        [0.0, 2.0, 2.0, 4.0, -0.0, -2.0, -2.0, 0.0, 1.0],
    ),
    (1, [0.25, 0.75, -0.25, -0.75, 1.125], [0.2, 0.8, -0.2, -0.8, 1.1]),
    (2, [0.125, 0.375, -0.125, -0.375], [0.12, 0.38, -0.12, -0.38]),
]


@pytest.mark.parametrize(("decimals", "data", "expected"), HALF_TIES)
def test_round_half_to_even(
    constructor: Constructor, decimals: int, data: list[float], expected: list[float]
) -> None:
    df = nw.from_native(constructor({"a": data}))
    result = df.select(nw.col("a").round(decimals))
    assert_equal_data(result, {"a": expected})


@pytest.mark.parametrize(("decimals", "data", "expected"), HALF_TIES)
def test_round_half_to_even_series(
    constructor_eager: ConstructorEager,
    decimals: int,
    data: list[float],
    expected: list[float],
) -> None:
    df = nw.from_native(constructor_eager({"a": data}), eager_only=True)
    assert_equal_data({"a": df["a"].round(decimals)}, {"a": expected})


@pytest.mark.parametrize("decimals", [0, 1])
def test_round_keeps_integer_dtype(
    constructor: Constructor, decimals: int, request: pytest.FixtureRequest
) -> None:
    if "ibis" in str(constructor) and decimals != 0:
        # Ibis returns `Float64` when rounding integers to a non-zero number of digits.
        request.applymarker(pytest.mark.xfail)
    is_duckdb_based = any(x in str(constructor) for x in ("duckdb", "sqlframe"))
    if is_duckdb_based and DUCKDB_VERSION < (1, 3):
        pytest.skip(reason="DuckDB<1.3 `round` returns `DOUBLE` for integer input")
    df = nw.from_native(constructor({"a": [1, -2, 3]}))
    result = df.select(nw.col("a").round(decimals))
    assert result.collect_schema()["a"] == nw.Int64
    assert_equal_data(result, {"a": [1, -2, 3]})


def test_round_keeps_large_integers_exact() -> None:
    # Correcting ties via a `DOUBLE` banker's-rounding function (DuckDB's
    # `round_even`) would return 9007199254740992 for the first value.
    duckdb = pytest.importorskip("duckdb")
    rel = duckdb.sql(
        "SELECT * FROM (VALUES (9007199254740993), (123456789012345678)) t(a)"
    )
    result = nw.from_native(rel).select(nw.col("a").round(0))
    assert result.collect_schema()["a"] == nw.Int64
    assert_equal_data(result, {"a": [9007199254740993, 123456789012345678]})


def test_round_half_to_even_duckdb_decimal() -> None:
    duckdb = pytest.importorskip("duckdb")
    if DUCKDB_VERSION < (1, 3):
        pytest.skip(reason="`cast_to_type` requires DuckDB 1.3")
    rel = duckdb.sql("SELECT * FROM (VALUES (0.5), (1.5), (2.5), (-0.5), (-2.5)) t(a)")
    result = nw.from_native(rel).select(nw.col("a").round(0))
    assert result.collect_schema()["a"] == nw.Decimal(precision=2, scale=0)
    assert_equal_data(
        result, {"a": [Decimal(0), Decimal(2), Decimal(2), Decimal(0), Decimal(-2)]}
    )
