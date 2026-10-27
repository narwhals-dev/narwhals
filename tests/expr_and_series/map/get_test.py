from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pytest
from packaging.version import Version as PkgVersion

import narwhals as nw
from tests.utils import PANDAS_VERSION, POLARS_VERSION, assert_equal_data

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager

string_data = [{"x": 1, "y": 2}, {"y": 3}, {}, None]
int_data = [{10: 1, 20: 2}, {20: 3}, {}, None]
expected = {"x": [1, None, None, None], "y": [2, 3, None, None]}


def map_frame(  # noqa: PLR0911
    request: pytest.FixtureRequest,
    constructor: Constructor,
    data: list[Any] = string_data,
) -> Any:
    """Build a frame with a `Map(String | Int64, Int64)` column `a` natively for each backend."""
    name = str(constructor)
    int_keys = data is int_data
    if any(backend in name for backend in ("dask", "modin", "cudf")):
        request.applymarker(pytest.mark.xfail(reason="no native Map dtype"))
        return constructor({"a": data})
    if "polars" in name and POLARS_VERSION < PkgVersion("2"):  # pragma: no cover
        pytest.skip(reason="native Map dtype requires Polars 2.0.0")
    if "pandas" in name and PANDAS_VERSION < PkgVersion("2.2"):  # pragma: no cover
        pytest.skip(reason="pandas too old for pyarrow types")
    pa = pytest.importorskip("pyarrow")

    key_type = pa.int64() if int_keys else pa.string()
    array = pa.array(data, type=pa.map_(key_type, pa.int64()))
    arrow_table = pa.table({"a": array})
    if "polars" in name:
        import polars as pl

        df = pl.DataFrame({"a": pl.Series(array)})
        return df.lazy() if "lazy" in name else df
    if "pandas" in name:
        import pandas as pd

        return arrow_table.to_pandas(types_mapper=pd.ArrowDtype)
    if "duckdb" in name:
        import duckdb

        return duckdb.sql("select * from arrow_table")
    if "ibis" in name:
        import ibis

        return ibis.memtable(arrow_table)
    if "sqlframe" in name:
        from sqlframe.base import types as sqlframe_types

        from tests.utils import sqlframe_session

        return sqlframe_session().createDataFrame(
            [(value,) for value in data], schema=_spark_schema(sqlframe_types, int_keys)
        )
    if "pyspark" in name:
        from pyspark.sql import types as pyspark_types

        from tests.utils import pyspark_session

        return pyspark_session().createDataFrame(
            [(value,) for value in data], schema=_spark_schema(pyspark_types, int_keys)
        )
    return arrow_table


def _spark_schema(types: Any, int_keys: bool) -> Any:  # noqa: FBT001
    key_type = types.LongType() if int_keys else types.StringType()
    map_type = types.MapType(key_type, types.LongType())
    return types.StructType([types.StructField("a", map_type)])


def test_get_expr(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    df = nw.from_native(map_frame(request, constructor))
    result = df.select(x=nw.col("a").map.get("x"), y=nw.col("a").map.get("y"))
    assert_equal_data(result, expected)


def test_get_series(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    df = nw.from_native(map_frame(request, constructor_eager), eager_only=True)
    result = {"x": df["a"].map.get("x"), "y": df["a"].map.get("y")}
    assert_equal_data(result, expected)


def test_get_int_key(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    df = nw.from_native(map_frame(request, constructor, int_data))
    result = df.select(x=nw.col("a").map.get(10), y=nw.col("a").map.get(20))
    assert_equal_data(result, expected)


def test_get_null_key(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    df = nw.from_native(map_frame(request, constructor))
    result = df.select(nw.col("a").map.get(None))
    assert_equal_data(result, {"a": [None, None, None, None]})


def test_get_duplicate_keys_pyarrow() -> None:
    pa = pytest.importorskip("pyarrow")
    array = pa.array([[("x", 1), ("x", 2)]], type=pa.map_(pa.string(), pa.int64()))
    df = nw.from_native(pa.table({"a": array}))
    assert_equal_data(df.select(nw.col("a").map.get("x")), {"a": [2]})


def test_get_pandas_non_pyarrow() -> None:
    pd = pytest.importorskip("pandas")
    s = nw.from_native(pd.Series([{"x": 1}]), series_only=True)
    msg = re.escape("Series must be of PyArrow Map type to support map namespace.")
    with pytest.raises(TypeError, match=msg):
        s.map.get("x")
