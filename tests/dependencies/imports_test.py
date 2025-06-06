from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals.exceptions import ModuleUpgradeRequiredError
from narwhals.utils import Implementation, import_optional_dependency

if TYPE_CHECKING:
    from narwhals.typing import IntoFrameT


@pytest.fixture
def data() -> dict[str, Any]:
    return {"a": [1, 1, 2], "b": [4, 5, 6]}


def _roundtrip_query(frame: IntoFrameT) -> IntoFrameT:
    return (
        nw.from_native(frame)
        .group_by("a")
        .agg(nw.col("b").mean())
        .filter(nw.col("a") > 1)
        .to_native()
    )


def test_import_polars(data: dict[str, Any]) -> None:
    # NOTE: Can't do `monkeypatch.delitem` safely
    #     ImportError: PyO3 modules compiled for CPython 3.8 or older may only be initialized once per interpreter process
    pytest.importorskip("polars")
    df = Implementation.POLARS.to_native_namespace().DataFrame(data)
    result = _roundtrip_query(df)
    import polars as pl

    assert isinstance(result, pl.DataFrame)


def test_import_dask(data: dict[str, Any]) -> None:
    pytest.importorskip("dask")
    df = Implementation.DASK.to_native_namespace().from_dict(data, npartitions=1)
    result = _roundtrip_query(df)
    import dask.dataframe as dd

    assert isinstance(result, dd.DataFrame)


def test_import_pandas(monkeypatch: pytest.MonkeyPatch, data: dict[str, Any]) -> None:
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


def test_import_pyarrow(monkeypatch: pytest.MonkeyPatch, data: dict[str, Any]) -> None:
    pytest.importorskip("pyarrow")
    monkeypatch.delitem(sys.modules, "pyarrow")
    df = Implementation.PYARROW.to_native_namespace().table(data)
    result = _roundtrip_query(df)
    import pyarrow as pa

    assert isinstance(result, pa.Table)


def test_module_not_found() -> None:
    with pytest.raises(ModuleNotFoundError, match="not_a_real_package"):
        import_optional_dependency(module_name="not_a_real_package")


def test_module_not_min_version() -> None:
    with pytest.raises(ModuleUpgradeRequiredError, match="requires min version"):
        import_optional_dependency(module_name="pandas", min_version=(999, 999, 999))
