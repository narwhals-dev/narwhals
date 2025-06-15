from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals._utils import Implementation
from narwhals.dependencies import get_cudf, get_modin, get_polars
from tests.utils import POLARS_VERSION, Constructor, assert_equal_data

if TYPE_CHECKING:
    from types import ModuleType


data = {"a": [1, 2], "b": [3, 4]}


def test_collect_to_default_backend(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    result = df.lazy().collect().to_native()

    expected_cls: Any
    if "polars" in str(constructor):
        pytest.importorskip("polars")
        import polars as pl

        expected_cls = pl.DataFrame
    elif any(x in str(constructor) for x in ("pandas", "dask")):
        pytest.importorskip("pandas")
        import pandas as pd

        expected_cls = pd.DataFrame
    elif "modin" in str(constructor):
        mpd = get_modin()
        expected_cls = mpd.DataFrame
    elif "cudf" in str(constructor):
        cudf = get_cudf()
        expected_cls = cudf.DataFrame
    else:  # pyarrow, duckdb, and PySpark
        pytest.importorskip("pyarrow")
        import pyarrow as pa

        expected_cls = pa.Table

    assert isinstance(result, expected_cls)


@pytest.mark.filterwarnings(
    "ignore:is_sparse is deprecated and will be removed in a future version."
)
@pytest.mark.parametrize("backend", ["pandas", Implementation.PANDAS])
def test_collect_to_valid_backend_pandas(
    constructor: Constructor, backend: Implementation | str | None
) -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = nw.from_native(constructor(data))
    result = df.lazy().collect(backend=backend).to_native()
    assert isinstance(result, pd.DataFrame)


@pytest.mark.filterwarnings(
    "ignore:is_sparse is deprecated and will be removed in a future version."
)
@pytest.mark.parametrize("backend", ["polars", Implementation.POLARS])
def test_collect_to_valid_backend_polars(
    constructor: Constructor, backend: Implementation | str | None
) -> None:
    pytest.importorskip("polars")
    import polars as pl

    df = nw.from_native(constructor(data))
    result = df.lazy().collect(backend=backend).to_native()
    assert isinstance(result, pl.DataFrame)


@pytest.mark.filterwarnings(
    "ignore:is_sparse is deprecated and will be removed in a future version."
)
@pytest.mark.parametrize("backend", ["pyarrow", Implementation.PYARROW])
def test_collect_to_valid_backend_pyarrow(
    constructor: Constructor, backend: Implementation | str | None
) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    df = nw.from_native(constructor(data))
    result = df.lazy().collect(backend=backend).to_native()
    assert isinstance(result, pa.Table)


@pytest.mark.filterwarnings(
    "ignore:is_sparse is deprecated and will be removed in a future version."
)
def test_collect_to_valid_backend_pandas_mod(constructor: Constructor) -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = nw.from_native(constructor(data))
    result = df.lazy().collect(backend=pd).to_native()
    assert isinstance(result, pd.DataFrame)


@pytest.mark.filterwarnings(
    "ignore:is_sparse is deprecated and will be removed in a future version."
)
def test_collect_to_valid_backend_polars_mod(constructor: Constructor) -> None:
    pytest.importorskip("polars")
    import polars as pl

    df = nw.from_native(constructor(data))
    result = df.lazy().collect(backend=pl).to_native()
    assert isinstance(result, pl.DataFrame)


@pytest.mark.filterwarnings(
    "ignore:is_sparse is deprecated and will be removed in a future version."
)
def test_collect_to_valid_backend_pyarrow_mod(constructor: Constructor) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    df = nw.from_native(constructor(data))
    result = df.lazy().collect(backend=pa).to_native()
    assert isinstance(result, pa.Table)


@pytest.mark.parametrize(
    "backend", ["foo", Implementation.DASK, Implementation.MODIN, pytest]
)
def test_collect_to_invalid_backend(
    constructor: Constructor, backend: ModuleType | Implementation | str | None
) -> None:
    df = nw.from_native(constructor(data))

    with pytest.raises(ValueError, match="Unsupported `backend` value"):
        df.lazy().collect(backend=backend).to_native()


def test_collect_with_kwargs(constructor: Constructor) -> None:
    pl_kwargs = (
        {"optimizations": get_polars().QueryOptFlags(predicate_pushdown=False)}
        if POLARS_VERSION > (1, 29, 0)
        else {"no_optimization": True}
    )
    collect_kwargs = {
        nw.Implementation.POLARS: pl_kwargs,
        nw.Implementation.DASK: {"optimize_graph": False},
        nw.Implementation.PYARROW: {},
    }

    df = nw.from_native(constructor(data))

    result = (
        df.lazy()
        .select(nw.col("a", "b").sum())
        .collect(**collect_kwargs.get(df.implementation, {}))
    )

    expected = {"a": [3], "b": [7]}
    assert_equal_data(result, expected)


def test_collect_empty(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3]}))
    lf = df.filter(nw.col("a").is_null()).with_columns(b=nw.lit(None)).lazy()
    result = lf.collect()
    assert_equal_data(result, {"a": [], "b": []})
