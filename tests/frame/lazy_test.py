from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals._utils import Implementation
from narwhals.dependencies import get_cudf, get_modin
from narwhals.testing.constructors import pyspark_session, sqlframe_session
from tests.utils import DUCKDB_VERSION, PANDAS_VERSION, assert_equal_data

if TYPE_CHECKING:
    from narwhals._typing import LazyAllowed, SparkLike
    from tests.utils import ConstructorEager


data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}


def test_lazy_to_default(nw_eager_constructor: ConstructorEager) -> None:
    df = nw.from_native(nw_eager_constructor(data), eager_only=True)
    result = df.lazy()
    assert isinstance(result, nw.LazyFrame)
    df = nw.from_native(nw_eager_constructor(data), eager_only=True)
    result = df.lazy()
    assert isinstance(result, nw.LazyFrame)

    expected_cls: Any
    if "polars" in str(nw_eager_constructor):
        import polars as pl

        expected_cls = pl.LazyFrame
    elif "pandas" in str(nw_eager_constructor):
        import pandas as pd

        expected_cls = pd.DataFrame
    elif "modin" in str(nw_eager_constructor):
        mpd = get_modin()
        expected_cls = mpd.DataFrame
    elif "cudf" in str(nw_eager_constructor):
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
    nw_eager_constructor: ConstructorEager,
    backend: LazyAllowed,
    request: pytest.FixtureRequest,
) -> None:
    impl = Implementation.from_backend(backend)
    pytest.importorskip(impl.name.lower())
    if (
        "pandas_constructor" in str(nw_eager_constructor)
        and impl.is_duckdb()
        and PANDAS_VERSION >= (3,)
        and DUCKDB_VERSION < (1, 4, 4)
    ):  # pragma: no cover
        # https://github.com/duckdb/duckdb/issues/18297
        request.applymarker(pytest.mark.xfail)
    if "pandas_nullable" in str(nw_eager_constructor):
        pytest.importorskip("pyarrow")

    is_spark_connect = os.environ.get("SPARK_CONNECT", None)
    if is_spark_connect is not None and impl.is_pyspark():  # pragma: no cover
        # Workaround for impl.name.lower() being "pyspark[connect]" for
        # Implementation.PYSPARK_CONNECT, which is never installed.
        impl = Implementation.PYSPARK_CONNECT

    df = nw.from_native(nw_eager_constructor(data), eager_only=True)
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
    nw_eager_constructor: ConstructorEager, backend: SparkLike
) -> None:
    impl = Implementation.from_backend(backend)
    pytest.importorskip(impl.name.lower())

    df = nw.from_native(nw_eager_constructor(data), eager_only=True)

    err_msg = re.escape("Spark like backends require `session` to be not None.")
    with pytest.raises(ValueError, match=err_msg):
        df.lazy(backend=backend, session=None)


def test_lazy_backend_invalid(nw_eager_constructor: ConstructorEager) -> None:
    df = nw.from_native(nw_eager_constructor(data), eager_only=True)
    with pytest.raises(ValueError, match="Not-supported backend"):
        df.lazy(backend=Implementation.PANDAS)  # type: ignore[arg-type]
