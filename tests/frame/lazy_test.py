from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from narwhals.dependencies import get_cudf
from narwhals.dependencies import get_modin
from narwhals.utils import Implementation

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


data = {"a": [1, 2, 3]}


def test_lazy_to_default(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.lazy()
    assert isinstance(result, nw.LazyFrame)
    df = nw_v1.from_native(constructor_eager(data), eager_only=True)
    result = df.lazy()
    assert isinstance(result, nw_v1.LazyFrame)

    if "polars" in str(constructor_eager):
        expected_cls = pl.LazyFrame
    elif "pandas" in str(constructor_eager):
        expected_cls = pd.DataFrame
    elif "modin" in str(constructor_eager):
        mpd = get_modin()
        expected_cls = mpd.DataFrame
    elif "cudf" in str(constructor_eager):
        cudf = get_cudf()
        expected_cls = cudf.DataFrame
    else:  # pyarrow
        expected_cls = pa.Table

    assert isinstance(result.to_native(), expected_cls)


@pytest.mark.parametrize(
    "backend",
    [
        Implementation.POLARS,
        Implementation.DUCKDB,
        Implementation.DASK,
        "polars",
        "duckdb",
        "dask",
    ],
)
def test_lazy_backend(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    backend: Implementation | str,
) -> None:
    if "modin" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
    if (backend is Implementation.DASK) or backend == "dask":
        pytest.importorskip("dask")
    if (backend is Implementation.DUCKDB) or backend == "duckdb":
        pytest.importorskip("duckdb")
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.lazy(backend=backend)
    assert isinstance(result, nw.LazyFrame)

    expected = (
        Implementation.from_string(backend) if isinstance(backend, str) else backend
    )
    assert result.implementation == expected


def test_lazy_backend_invalid(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    with pytest.raises(ValueError, match="Not-supported backend"):
        df.lazy(backend=Implementation.PANDAS)
