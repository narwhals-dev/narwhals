from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals._utils import Implementation
from narwhals.dependencies import get_cudf, get_modin
from tests.utils import (
    PANDAS_VERSION,
    assert_equal_data,
    pyspark_session,
    sqlframe_session,
)

if TYPE_CHECKING:
    from narwhals._typing import LazyAllowed, SparkLike
    from tests.utils import ConstructorEager


data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}


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


@pytest.mark.slow
@pytest.mark.parametrize(
    "backend",
    [
        Implementation.POLARS,
        Implementation.DUCKDB,
        Implementation.DASK,
        Implementation.IBIS,
        Implementation.PYSPARK,
        Implementation.SQLFRAME,
        "polars",
        "duckdb",
        "dask",
        "ibis",
        "pyspark",
        "sqlframe",
    ],
)
def test_lazy(
    constructor_eager: ConstructorEager,
    backend: LazyAllowed,
    request: pytest.FixtureRequest,
) -> None:
    impl = Implementation.from_backend(backend)
    pytest.importorskip(impl.name.lower())
    if (
        "pandas_constructor" in str(constructor_eager)
        and impl.is_duckdb()
        and PANDAS_VERSION >= (3,)
    ):  # pragma: no cover
        # https://github.com/duckdb/duckdb/issues/18297
        request.applymarker(pytest.mark.xfail)
    if "pandas_nullable" in str(constructor_eager):
        pytest.importorskip("pyarrow")

    is_spark_connect = os.environ.get("SPARK_CONNECT", None)
    if is_spark_connect is not None and impl.is_pyspark():  # pragma: no cover
        # Workaround for impl.name.lower() being "pyspark[connect]" for
        # Implementation.PYSPARK_CONNECT, which is never installed.
        impl = Implementation.PYSPARK_CONNECT

    df = nw.from_native(constructor_eager(data), eager_only=True)
    session: Any
    if impl.is_sqlframe():
        session = sqlframe_session()
    elif impl.is_pyspark() or impl.is_pyspark_connect():  # pragma: no cover
        session = pyspark_session()
    else:
        session = None

    result = df.lazy(backend=backend, session=session)
    assert isinstance(result, nw.LazyFrame)
    assert result.implementation == impl
    assert_equal_data(df.sort("a"), data)


@pytest.mark.parametrize("backend", ["pyspark", "sqlframe"])
def test_lazy_spark_like_requires_session(
    constructor_eager: ConstructorEager, backend: SparkLike
) -> None:
    impl = Implementation.from_backend(backend)
    pytest.importorskip(impl.name.lower())

    df = nw.from_native(constructor_eager(data), eager_only=True)

    err_msg = re.escape("Spark like backends require `session` to be not None.")
    with pytest.raises(ValueError, match=err_msg):
        df.lazy(backend=backend, session=None)


def test_lazy_backend_invalid(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    with pytest.raises(ValueError, match="Not-supported backend"):
        df.lazy(backend=Implementation.PANDAS)  # type: ignore[arg-type]
