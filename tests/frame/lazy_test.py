from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals._utils import Implementation
from narwhals.dependencies import get_cudf, get_modin

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


data = {"a": [1, 2, 3]}


def test_lazy_to_default(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.lazy()
    assert isinstance(result, nw.LazyFrame)
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.lazy()
    assert isinstance(result, nw.LazyFrame)

    expected_cls: Any
    if "polars" in str(constructor_eager):
        import polars as pl

        expected_cls = pl.LazyFrame
    elif "pandas" in str(constructor_eager):
        import pandas as pd

        expected_cls = pd.DataFrame
    elif "modin" in str(constructor_eager):
        mpd = get_modin()
        expected_cls = mpd.DataFrame
    elif "cudf" in str(constructor_eager):
        cudf = get_cudf()
        expected_cls = cudf.DataFrame
    else:  # pyarrow
        import pyarrow as pa

        expected_cls = pa.Table

    assert isinstance(result.to_native(), expected_cls)


@pytest.mark.parametrize(
    "backend",
    [
        Implementation.POLARS,
        Implementation.DUCKDB,
        Implementation.DASK,
        Implementation.IBIS,
        "polars",
        "duckdb",
        "dask",
        "ibis",
    ],
)
def test_lazy_backend(
    constructor_eager: ConstructorEager, backend: Implementation | str
) -> None:
    implementation = Implementation.from_backend(backend)
    pytest.importorskip(implementation.name.lower())
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.lazy(backend=backend)
    assert isinstance(result, nw.LazyFrame)
    assert result.implementation == implementation


def test_lazy_backend_invalid(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    with pytest.raises(ValueError, match="Not-supported backend"):
        df.lazy(backend=Implementation.PANDAS)
