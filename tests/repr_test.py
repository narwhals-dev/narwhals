from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION


def test_repr(request: pytest.FixtureRequest) -> None:
    pytest.importorskip("duckdb")
    pytest.importorskip("pandas")
    import duckdb
    import pandas as pd

    if PANDAS_VERSION >= (3,):
        # https://github.com/duckdb/duckdb/issues/18297
        request.applymarker(pytest.mark.xfail)

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["fdaf", "fda", "cf"]})
    result = nw.from_native(df).__repr__()
    expected = (
        "┌──────────────────┐\n"
        "|Narwhals DataFrame|\n"
        "|------------------|\n"
        "|       a     b    |\n"
        "|    0  1  fdaf    |\n"
        "|    1  2   fda    |\n"
        "|    2  3    cf    |\n"
        "└──────────────────┘"
    )
    assert result == expected
    result = nw.from_native(df).lazy().__repr__()
    expected = (
        "┌──────────────────┐\n"
        "|Narwhals LazyFrame|\n"
        "|------------------|\n"
        "|       a     b    |\n"
        "|    0  1  fdaf    |\n"
        "|    1  2   fda    |\n"
        "|    2  3    cf    |\n"
        "└──────────────────┘"
    )
    assert result == expected
    result = nw.from_native(df)["a"].__repr__()
    expected = (
        "┌─────────────────────┐\n"
        "|   Narwhals Series   |\n"
        "|---------------------|\n"
        "|0    1               |\n"
        "|1    2               |\n"
        "|2    3               |\n"
        "|Name: a, dtype: int64|\n"
        "└─────────────────────┘"
    )
    assert result == expected
    result = nw.from_native(duckdb.table("df")).__repr__()
    expected = (
        "┌───────────────────┐\n"
        "|Narwhals LazyFrame |\n"
        "|-------------------|\n"
        "|┌───────┬─────────┐|\n"
        "|│   a   │    b    │|\n"
        "|│ int64 │ varchar │|\n"
        "|├───────┼─────────┤|\n"
        "|│     1 │ fdaf    │|\n"
        "|│     2 │ fda     │|\n"
        "|│     3 │ cf      │|\n"
        "|└───────┴─────────┘|\n"
        "└───────────────────┘"
    )
    assert result == expected
    # Make something wider than the terminal size
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["fdaf" * 100, "fda", "cf"]})
    result = nw.from_native(duckdb.table("df")).__repr__()
    expected = (
        "┌───────────────────────────────────────┐\n"
        "|          Narwhals LazyFrame           |\n"
        "| Use `.to_native` to see native output |\n"
        "└───────────────────────────────────────┘"
    )
    assert result == expected


def test_polars_series_repr() -> None:
    pytest.importorskip("polars")
    import polars as pl

    df = nw.from_native(pl.DataFrame({"col1": [None, 2], "col2": [3, 7]}))
    s = df["col1"]
    result = repr(s)
    expected = (
        "┌────────────────────┐\n"
        "|  Narwhals Series   |\n"
        "|--------------------|\n"
        "|shape: (2,)         |\n"
        "|Series: 'col1' [i64]|\n"
        "|[                   |\n"
        "|        null        |\n"
        "|        2           |\n"
        "|]                   |\n"
        "└────────────────────┘"
    )
    assert result == expected


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (nw.col("a"), "col(a)"),
        (nw.col("a").abs(), "col(a).abs()"),
        (nw.col("a").std(ddof=2), "col(a).std(ddof=2)"),
        (
            nw.sum_horizontal(nw.col("a").rolling_mean(2), "b"),
            "sum_horizontal(col(a).rolling_mean(window_size=2, min_samples=2, center=False), b)",
        ),
    ],
)
def test_expr_repr(expr: nw.Expr, expected: str) -> None:
    assert repr(expr) == expected
