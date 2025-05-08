from __future__ import annotations

import sys

import pandas as pd
import pytest

import narwhals as nw


def test_polars(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("polars")
    import polars as pl

    monkeypatch.delitem(sys.modules, "pandas")
    monkeypatch.delitem(sys.modules, "numpy")
    monkeypatch.delitem(sys.modules, "pyarrow", raising=False)
    monkeypatch.delitem(sys.modules, "typing_extensions", raising=False)
    monkeypatch.delitem(sys.modules, "duckdb", raising=False)
    monkeypatch.delitem(sys.modules, "dask", raising=False)
    monkeypatch.delitem(sys.modules, "ibis", raising=False)
    monkeypatch.delitem(sys.modules, "pyspark", raising=False)
    df = pl.DataFrame({"a": [1, 1, 2], "b": [4, 5, 6]})
    nw.from_native(df, eager_only=True).group_by("a").agg(nw.col("b").mean()).filter(
        nw.col("a") > 1
    )
    assert "polars" in sys.modules
    assert "pandas" not in sys.modules
    assert "numpy" not in sys.modules
    assert "pyarrow" not in sys.modules
    assert "dask" not in sys.modules
    assert "ibis" not in sys.modules
    assert "pyspark" not in sys.modules
    assert "duckdb" not in sys.modules
    assert "typing_extensions" not in sys.modules


def test_pandas(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "polars", raising=False)
    monkeypatch.delitem(sys.modules, "pyarrow", raising=False)
    monkeypatch.delitem(sys.modules, "duckdb", raising=False)
    monkeypatch.delitem(sys.modules, "dask", raising=False)
    monkeypatch.delitem(sys.modules, "ibis", raising=False)
    monkeypatch.delitem(sys.modules, "pyspark", raising=False)
    df = pd.DataFrame({"a": [1, 1, 2], "b": [4, 5, 6]})
    nw.from_native(df, eager_only=True).group_by("a").agg(nw.col("b").mean()).filter(
        nw.col("a") > 1
    )
    assert "polars" not in sys.modules
    assert "pandas" in sys.modules
    assert "numpy" in sys.modules
    assert "pyarrow" not in sys.modules
    assert "dask" not in sys.modules
    assert "ibis" not in sys.modules
    assert "pyspark" not in sys.modules
    assert "duckdb" not in sys.modules


def test_dask(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("dask")
    import dask.dataframe as dd

    monkeypatch.delitem(sys.modules, "polars", raising=False)
    monkeypatch.delitem(sys.modules, "pyarrow", raising=False)
    monkeypatch.delitem(sys.modules, "duckdb", raising=False)
    monkeypatch.delitem(sys.modules, "pyspark", raising=False)
    df = dd.from_pandas(pd.DataFrame({"a": [1, 1, 2], "b": [4, 5, 6]}))
    nw.from_native(df).group_by("a").agg(nw.col("b").mean()).filter(nw.col("a") > 1)
    assert "polars" not in sys.modules
    assert "pandas" in sys.modules
    assert "numpy" in sys.modules
    assert "pyarrow" not in sys.modules
    assert "dask" in sys.modules
    assert "pyspark" not in sys.modules
    assert "duckdb" not in sys.modules


def test_pyarrow(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    monkeypatch.delitem(sys.modules, "polars", raising=False)
    monkeypatch.delitem(sys.modules, "pandas")
    monkeypatch.delitem(sys.modules, "duckdb", raising=False)
    monkeypatch.delitem(sys.modules, "dask", raising=False)
    monkeypatch.delitem(sys.modules, "ibis", raising=False)
    monkeypatch.delitem(sys.modules, "pyspark", raising=False)
    df = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    nw.from_native(df).group_by("a").agg(nw.col("b").mean()).filter(nw.col("a") > 1)
    assert "polars" not in sys.modules
    assert "pandas" not in sys.modules
    assert "numpy" in sys.modules
    assert "pyarrow" in sys.modules
    assert "dask" not in sys.modules
    assert "ibis" not in sys.modules
    assert "pyspark" not in sys.modules
    assert "duckdb" not in sys.modules
