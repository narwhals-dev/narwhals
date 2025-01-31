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
from tests.utils import PANDAS_VERSION
from tests.utils import Constructor
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from types import ModuleType


data = {"a": [1, 2], "b": [3, 4]}


def test_collect_to_default_backend(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.lazy().collect().to_native()

    if "polars" in str(constructor):
        expected_cls = pl.DataFrame
    elif any(x in str(constructor) for x in ("pandas", "dask", "pyspark")):
        expected_cls = pd.DataFrame
    elif "modin" in str(constructor):
        mpd = get_modin()
        expected_cls = mpd.DataFrame
    elif "cudf" in str(constructor):
        cudf = get_cudf()
        expected_cls = cudf.DataFrame
    else:  # pyarrow and duckdb
        expected_cls = pa.Table

    assert isinstance(result, expected_cls)


@pytest.mark.filterwarnings(
    "ignore:is_sparse is deprecated and will be removed in a future version."
)
@pytest.mark.parametrize(
    ("backend", "expected_cls"),
    [
        ("pyarrow", pa.Table),
        ("polars", pl.DataFrame),
        ("pandas", pd.DataFrame),
        (Implementation.PYARROW, pa.Table),
        (Implementation.POLARS, pl.DataFrame),
        (Implementation.PANDAS, pd.DataFrame),
        (pa, pa.Table),
        (pl, pl.DataFrame),
        (pd, pd.DataFrame),
    ],
)
def test_collect_to_valid_backend(
    constructor: Constructor,
    backend: ModuleType | Implementation | str | None,
    expected_cls: type,
    request: pytest.FixtureRequest,
) -> None:
    if "pandas" in str(constructor) and PANDAS_VERSION < (1,):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.lazy().collect(backend=backend).to_native()
    assert isinstance(result, expected_cls)


@pytest.mark.parametrize(
    "backend", ["foo", Implementation.DASK, Implementation.MODIN, pytest]
)
def test_collect_to_invalid_backend(
    constructor: Constructor,
    backend: ModuleType | Implementation | str | None,
) -> None:
    df = nw.from_native(constructor(data))

    with pytest.raises(ValueError, match="Unsupported `backend` value"):
        df.lazy().collect(backend=backend).to_native()


def test_collect_with_kwargs(constructor: Constructor) -> None:
    collect_kwargs = {
        nw.Implementation.POLARS: {"no_optimization": True},
        nw.Implementation.DASK: {"optimize_graph": False},
        nw.Implementation.PYARROW: {},
    }

    df = nw_v1.from_native(constructor(data))

    result = (
        df.lazy()
        .select(nw_v1.col("a", "b").sum())
        .collect(**collect_kwargs.get(df.implementation, {}))  # type: ignore[arg-type]
    )

    expected = {"a": [3], "b": [7]}
    assert_equal_data(result, expected)
