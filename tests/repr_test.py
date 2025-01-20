from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw

pytest.importorskip("duckdb")
pytest.importorskip("pandas")
import duckdb  # noqa: E402
import pandas as pd  # noqa: E402


def test_repr() -> None:
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
        "|Narwhals DataFrame |\n"
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
        "|          Narwhals DataFrame           |\n"
        "| Use `.to_native` to see native output |\n"
        "└───────────────────────────────────────┘"
    )
    assert result == expected
