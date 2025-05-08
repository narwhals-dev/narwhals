from __future__ import annotations

import pytest

import narwhals as nw


def test_implementation_pandas() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    assert (
        nw.from_native(pd.DataFrame({"a": [1, 2, 3]})).implementation
        is nw.Implementation.PANDAS
    )
    assert (
        nw.from_native(pd.DataFrame({"a": [1, 2, 3]}))["a"].implementation
        is nw.Implementation.PANDAS
    )
    assert nw.from_native(pd.DataFrame({"a": [1, 2, 3]})).implementation.is_pandas()
    assert nw.from_native(pd.DataFrame({"a": [1, 2, 3]})).implementation.is_pandas_like()


def test_implementation_polars() -> None:
    pytest.importorskip("polars")
    import polars as pl

    assert not nw.from_native(pl.DataFrame({"a": [1, 2, 3]})).implementation.is_pandas()
    assert not nw.from_native(pl.DataFrame({"a": [1, 2, 3]}))[
        "a"
    ].implementation.is_pandas()
    assert nw.from_native(pl.DataFrame({"a": [1, 2, 3]})).implementation.is_polars()
    assert nw.from_native(pl.LazyFrame({"a": [1, 2, 3]})).implementation.is_polars()


@pytest.mark.parametrize(
    ("member", "value"),
    [
        ("PANDAS", "pandas"),
        ("MODIN", "modin"),
        ("CUDF", "cudf"),
        ("PYARROW", "pyarrow"),
        ("PYSPARK", "pyspark"),
        ("POLARS", "polars"),
        ("DASK", "dask"),
        ("DUCKDB", "duckdb"),
        ("IBIS", "ibis"),
        ("SQLFRAME", "sqlframe"),
        ("PYSPARK_CONNECT", "pyspark[connect]"),
        ("UNKNOWN", "unknown"),
    ],
)
def test_implementation_new(member: str, value: str) -> None:
    assert nw.Implementation(value) is getattr(nw.Implementation, member)
