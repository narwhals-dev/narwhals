from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager


data = {"a": [1, 2, 3]}


def _get_expected_namespace(constructor_name: str) -> Any | None:  # noqa: PLR0911
    """Get expected namespace module for a given constructor."""
    if "pandas" in constructor_name:
        import pandas as pd

        return pd
    if "polars" in constructor_name:
        pytest.importorskip("polars")
        import polars as pl

        return pl
    if "pyarrow_table" in constructor_name:
        pytest.importorskip("pyarrow")
        import pyarrow as pa

        return pa
    if "duckdb" in constructor_name:
        pytest.importorskip("duckdb")
        import duckdb

        return duckdb
    if "cudf" in constructor_name:  # pragma: no cover
        pytest.importorskip("cudf")
        import cudf

        return cudf
    if "modin" in constructor_name:
        pytest.importorskip("modin")
        import modin.pandas as mpd

        return mpd
    if "dask" in constructor_name:
        pytest.importorskip("dask")
        import dask.dataframe as dd

        return dd
    if "ibis" in constructor_name:
        pytest.importorskip("ibis")
        import ibis

        return ibis
    if "sqlframe" in constructor_name:
        pytest.importorskip("sqlframe")
        import sqlframe

        return sqlframe
    return None  # pragma: no cover


def test_native_namespace_frame(constructor: Constructor) -> None:
    constructor_name = str(constructor)
    if "pyspark" in constructor_name and "sqlframe" not in constructor_name:
        pytest.skip(reason="Requires special handling for spark local vs spark connect")

    expected_namespace = _get_expected_namespace(constructor_name=constructor_name)

    df = nw.from_native(constructor(data))
    assert nw.get_native_namespace(df) is expected_namespace
    assert nw.get_native_namespace(df.to_native()) is expected_namespace
    assert nw.get_native_namespace(df.lazy().to_native()) is expected_namespace


def test_native_namespace_series(constructor_eager: ConstructorEager) -> None:
    constructor_name = constructor_eager.__name__

    expected_namespace = _get_expected_namespace(constructor_name=constructor_name)

    df = nw.from_native(constructor_eager(data), eager_only=True)

    assert nw.get_native_namespace(df["a"].to_native()) is expected_namespace
    assert nw.get_native_namespace(df, df["a"].to_native()) is expected_namespace


def test_get_native_namespace_invalid() -> None:
    with pytest.raises(TypeError, match="Unsupported type: 'int'"):
        nw.get_native_namespace(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="At least one object"):
        nw.get_native_namespace()


def test_get_native_namespace_invalid_cross() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("polars")

    import pandas as pd
    import polars as pl

    with pytest.raises(ValueError, match="Found objects with different"):
        nw.get_native_namespace(pd.Series([1]), pl.Series([2]))
