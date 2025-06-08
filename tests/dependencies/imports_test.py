from __future__ import annotations

import sys
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals.utils import Implementation

if TYPE_CHECKING:
    from narwhals.typing import IntoFrameT


data = {"a": [1, 1, 2], "b": [4, 5, 6]}


def _roundtrip_query(frame: IntoFrameT) -> IntoFrameT:
    return (
        nw.from_native(frame)
        .group_by("a")
        .agg(nw.col("b").mean())
        .filter(nw.col("a") > 1)
        .to_native()
    )


def test_import_polars() -> None:
    # NOTE: Can't do `monkeypatch.delitem` safely
    #     ImportError: PyO3 modules compiled for CPython 3.8 or older may only be initialized once per interpreter process
    pytest.importorskip("polars")
    df = Implementation.POLARS.to_native_namespace().DataFrame(data)
    result = _roundtrip_query(df)
    import polars as pl

    assert isinstance(result, pl.DataFrame)


def test_import_dask() -> None:
    pytest.importorskip("dask")
    df = Implementation.DASK.to_native_namespace().from_dict(data, npartitions=1)
    result = _roundtrip_query(df)
    import dask.dataframe as dd

    assert isinstance(result, dd.DataFrame)


def test_import_pandas(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pandas")
    if sys.version_info >= (3, 9):
        monkeypatch.delitem(sys.modules, "pandas")
    else:  # pragma: no cover
        # NOTE: AttributeError: partially initialized module 'pandas' has no attribute 'compat' (most likely due to a circular import)
        ...
    df = Implementation.PANDAS.to_native_namespace().DataFrame(data)
    result = _roundtrip_query(df)
    import pandas as pd

    assert isinstance(result, pd.DataFrame)


def test_import_pyarrow(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pyarrow")
    monkeypatch.delitem(sys.modules, "pyarrow")
    df = Implementation.PYARROW.to_native_namespace().table(data)
    result = _roundtrip_query(df)
    import pyarrow as pa

    assert isinstance(result, pa.Table)


@pytest.mark.parametrize(
    "impl",
    [
        Implementation.CUDF,
        Implementation.DASK,
        Implementation.DUCKDB,
        Implementation.IBIS,
        Implementation.MODIN,
        Implementation.PANDAS,
        Implementation.POLARS,
        Implementation.PYARROW,
        Implementation.PYSPARK,
        Implementation.SQLFRAME,
    ],
)
def test_to_native_namespace(
    monkeypatch: pytest.MonkeyPatch, impl: Implementation
) -> None:
    pytest.importorskip(impl.value)

    assert isinstance(impl.to_native_namespace(), ModuleType)

    monkeypatch.setattr(
        "narwhals._utils.Implementation._backend_version", lambda _: (0, 0, 1)
    )

    with pytest.raises(ValueError, match="Minimum version"):
        impl.to_native_namespace()


def test_to_native_namespace_unknown() -> None:
    impl = Implementation.UNKNOWN
    with pytest.raises(
        AssertionError, match="Cannot return native namespace from UNKNOWN Implementation"
    ):
        impl.to_native_namespace()
