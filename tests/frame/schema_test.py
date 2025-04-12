from __future__ import annotations

import re
from datetime import date
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import TYPE_CHECKING
from typing import Any

import pandas as pd
import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
    from collections.abc import Sequence

    from narwhals.typing import DTypeBackend
    from tests.utils import Constructor
    from tests.utils import ConstructorEager


data = {
    "a": [datetime(2020, 1, 1)],
    "b": [datetime(2020, 1, 1, tzinfo=timezone.utc)],
}


@pytest.mark.filterwarnings("ignore:Determining|Resolving.*")
def test_schema(constructor: Constructor) -> None:
    df = nw_v1.from_native(
        constructor({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8.0, 9.0]})
    )
    result = df.schema
    expected = {"a": nw_v1.Int64, "b": nw_v1.Int64, "z": nw_v1.Float64}

    result = df.schema
    assert result == expected
    result = df.lazy().collect().schema
    assert result == expected


def test_collect_schema(constructor: Constructor) -> None:
    df = nw_v1.from_native(
        constructor({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8.0, 9.0]})
    )
    expected = {"a": nw_v1.Int64, "b": nw_v1.Int64, "z": nw_v1.Float64}

    result = df.collect_schema()
    assert result == expected
    result = df.lazy().collect().collect_schema()
    assert result == expected


def test_schema_comparison() -> None:
    assert {"a": nw_v1.String()} != {"a": nw_v1.Int32()}
    assert {"a": nw_v1.Int32()} == {"a": nw_v1.Int32()}


def test_object() -> None:
    class Foo: ...

    df = pd.DataFrame({"a": [Foo()]}).astype(object)
    result = nw_v1.from_native(df).schema
    assert result["a"] == nw_v1.Object


def test_string_disguised_as_object() -> None:
    df = pd.DataFrame({"a": ["foo", "bar"]}).astype(object)
    result = nw_v1.from_native(df).schema
    assert result["a"] == nw_v1.String


def test_actual_object(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if any(x in str(constructor_eager) for x in ("pyarrow_table", "cudf")):
        request.applymarker(pytest.mark.xfail)

    class Foo: ...

    data = {"a": [Foo()]}
    df = nw_v1.from_native(constructor_eager(data))
    result = df.schema
    assert result == {"a": nw_v1.Object}


@pytest.mark.skipif(PANDAS_VERSION < (2, 0, 0), reason="too old")
def test_dtypes() -> None:
    pytest.importorskip("polars")
    import polars as pl

    df_pl = pl.DataFrame(
        {
            "a": [1],
            "b": [1],
            "c": [1],
            "d": [1],
            "e": [1],
            "f": [1],
            "g": [1],
            "h": [1],
            "i": [1],
            "j": [1],
            "k": ["1"],
            "l": [1],
            "m": [True],
            "n": [date(2020, 1, 1)],
            "o": [datetime(2020, 1, 1)],
            "p": ["a"],
            "q": [timedelta(1)],
            "r": ["a"],
            "s": [[1, 2]],
            "u": [{"a": 1}],
        },
        schema={
            "a": pl.Int64,
            "b": pl.Int32,
            "c": pl.Int16,
            "d": pl.Int8,
            "e": pl.UInt64,
            "f": pl.UInt32,
            "g": pl.UInt16,
            "h": pl.UInt8,
            "i": pl.Float64,
            "j": pl.Float32,
            "k": pl.String,
            "l": pl.Datetime,
            "m": pl.Boolean,
            "n": pl.Date,
            "o": pl.Datetime,
            "p": pl.Categorical,
            "q": pl.Duration,
            "r": pl.Enum(["a", "b"]),
            "s": pl.List(pl.Int64),
            "u": pl.Struct({"a": pl.Int64}),
        },
    )
    df_from_pl = nw_v1.from_native(df_pl, eager_only=True)
    expected = {
        "a": nw_v1.Int64,
        "b": nw_v1.Int32,
        "c": nw_v1.Int16,
        "d": nw_v1.Int8,
        "e": nw_v1.UInt64,
        "f": nw_v1.UInt32,
        "g": nw_v1.UInt16,
        "h": nw_v1.UInt8,
        "i": nw_v1.Float64,
        "j": nw_v1.Float32,
        "k": nw_v1.String,
        "l": nw_v1.Datetime,
        "m": nw_v1.Boolean,
        "n": nw_v1.Date,
        "o": nw_v1.Datetime,
        "p": nw_v1.Categorical,
        "q": nw_v1.Duration,
        "r": nw_v1.Enum,
        "s": nw_v1.List,
        "u": nw_v1.Struct,
    }

    assert df_from_pl.schema == df_from_pl.collect_schema()
    assert df_from_pl.schema == expected
    assert {name: df_from_pl[name].dtype for name in df_from_pl.columns} == expected

    # pandas/pyarrow only have categorical, not enum
    expected["r"] = nw_v1.Categorical

    df_from_pd = nw_v1.from_native(
        df_pl.to_pandas(use_pyarrow_extension_array=True), eager_only=True
    )

    assert df_from_pd.schema == df_from_pd.collect_schema() == expected
    assert {name: df_from_pd[name].dtype for name in df_from_pd.columns} == expected

    df_from_pa = nw_v1.from_native(df_pl.to_arrow(), eager_only=True)

    assert df_from_pa.schema == df_from_pa.collect_schema() == expected
    assert {name: df_from_pa[name].dtype for name in df_from_pa.columns} == expected


def test_unknown_dtype() -> None:
    df = pd.DataFrame({"a": pd.period_range("2000", periods=3, freq="M")})
    assert nw_v1.from_native(df).schema == {"a": nw_v1.Unknown}


def test_hash() -> None:
    assert nw_v1.Int64() in {nw_v1.Int64, nw_v1.Int32}


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("names", ["a", "b", "c"]),
        ("dtypes", [nw_v1.Int64(), nw_v1.Float32(), nw_v1.String()]),
        ("len", 3),
    ],
)
def test_schema_object(method: str, expected: Any) -> None:
    data = {"a": nw_v1.Int64(), "b": nw_v1.Float32(), "c": nw_v1.String()}
    schema = nw_v1.Schema(data)
    assert getattr(schema, method)() == expected


def test_validate_not_duplicated_columns_pandas_like() -> None:
    df = pd.DataFrame([[1, 2], [4, 5]], columns=["a", "a"])
    with pytest.raises(
        ValueError, match="Expected unique column names, got:\n- 'a' 2 times"
    ):
        nw_v1.from_native(df, eager_only=True)


def test_validate_not_duplicated_columns_arrow() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    table = pa.Table.from_arrays([pa.array([1, 2]), pa.array([4, 5])], names=["a", "a"])
    with pytest.raises(
        ValueError, match="Expected unique column names, got:\n- 'a' 2 times"
    ):
        nw_v1.from_native(table, eager_only=True)


def test_validate_not_duplicated_columns_duckdb() -> None:
    pytest.importorskip("duckdb")
    import duckdb

    rel = duckdb.sql("SELECT 1 AS a, 2 AS a")
    with pytest.raises(
        ValueError, match="Expected unique column names, got:\n- 'a' 2 times"
    ):
        nw_v1.from_native(rel, eager_only=False).lazy().collect()


@pytest.mark.skipif(
    PANDAS_VERSION < (2, 2, 0),
    reason="too old for pyarrow types",
)
def test_nested_dtypes() -> None:
    pytest.importorskip("duckdb")
    pytest.importorskip("polars")

    import duckdb
    import polars as pl

    df_pd = pl.DataFrame(
        {"a": [[1, 2]], "b": [[1, 2]], "c": [{"a": 1}]},
        schema_overrides={"b": pl.Array(pl.Int64, 2)},
    ).to_pandas(use_pyarrow_extension_array=True)
    nwdf: nw_v1.DataFrame[Any] | nw_v1.LazyFrame[Any] = nw_v1.from_native(df_pd)
    assert nwdf.schema == {
        "a": nw_v1.List(nw_v1.Int64),
        "b": nw_v1.Array(nw_v1.Int64, 2),
        "c": nw_v1.Struct({"a": nw_v1.Int64}),
    }
    df_pl = pl.DataFrame(
        {"a": [[1, 2]], "b": [[1, 2]], "c": [{"a": 1}]},
        schema_overrides={"b": pl.Array(pl.Int64, 2)},
    )
    nwdf = nw_v1.from_native(df_pl)
    assert nwdf.schema == {
        "a": nw_v1.List(nw_v1.Int64),
        "b": nw_v1.Array(nw_v1.Int64, 2),
        "c": nw_v1.Struct({"a": nw_v1.Int64}),
    }

    df_pa = pl.DataFrame(
        {"a": [[1, 2]], "b": [[1, 2]], "c": [{"a": 1, "b": "x", "c": 1.1}]},
        schema_overrides={"b": pl.Array(pl.Int64, 2)},
    ).to_arrow()
    nwdf = nw_v1.from_native(df_pa)
    assert nwdf.schema == {
        "a": nw_v1.List(nw_v1.Int64),
        "b": nw_v1.Array(nw_v1.Int64, 2),
        "c": nw_v1.Struct({"a": nw_v1.Int64, "b": nw_v1.String, "c": nw_v1.Float64}),
    }
    rel = duckdb.sql("select * from df_pa")
    nwdf = nw_v1.from_native(rel)
    assert nwdf.schema == {
        "a": nw_v1.List(nw_v1.Int64),
        "b": nw_v1.Array(nw_v1.Int64, 2),
        "c": nw_v1.Struct({"a": nw_v1.Int64, "b": nw_v1.String, "c": nw_v1.Float64}),
    }


def test_nested_dtypes_ibis(request: pytest.FixtureRequest) -> None:  # pragma: no cover
    pytest.importorskip("ibis")
    pytest.importorskip("polars")

    import ibis
    import polars as pl

    if PANDAS_VERSION < (1, 1):
        request.applymarker(pytest.mark.xfail)
    df = pl.DataFrame(
        {"a": [[1, 2]], "b": [[1, 2]], "c": [{"a": 1}]},
        schema_overrides={"b": pl.Array(pl.Int64, 2)},
    )
    tbl = ibis.memtable(df[["a", "c"]])
    nwdf = nw_v1.from_native(tbl)
    assert nwdf.schema == {
        "a": nw_v1.List(nw_v1.Int64),
        "c": nw_v1.Struct({"a": nw_v1.Int64}),
    }


@pytest.mark.skipif(
    PANDAS_VERSION < (2, 2, 0),
    reason="too old for pyarrow types",
)
def test_nested_dtypes_dask() -> None:
    pytest.importorskip("dask")
    pytest.importorskip("polars")

    import dask.dataframe as dd
    import polars as pl

    df = dd.from_pandas(
        pl.DataFrame(
            {"a": [[1, 2]], "b": [[1, 2]], "c": [{"a": 1}]},
            schema_overrides={"b": pl.Array(pl.Int64, 2)},
        ).to_pandas(use_pyarrow_extension_array=True)
    )
    nwdf = nw_v1.from_native(df)
    assert nwdf.schema == {
        "a": nw_v1.List(nw_v1.Int64),
        "b": nw_v1.Array(nw_v1.Int64, 2),
        "c": nw_v1.Struct({"a": nw_v1.Int64}),
    }


def test_all_nulls_pandas() -> None:
    assert (
        nw.from_native(pd.Series([None] * 3, dtype="object"), series_only=True).dtype
        == nw.String
    )
    assert (
        nw_v1.from_native(pd.Series([None] * 3, dtype="object"), series_only=True).dtype
        == nw_v1.Object
    )


@pytest.mark.parametrize(
    ("dtype_backend", "expected"),
    [
        (
            None,
            {"a": "int64", "b": str, "c": "bool", "d": "float64", "e": "datetime64[ns]"},
        ),
        (
            "pyarrow",
            {
                "a": "Int64[pyarrow]",
                "b": "string[pyarrow]",
                "c": "boolean[pyarrow]",
                "d": "Float64[pyarrow]",
                "e": "timestamp[ns][pyarrow]",
            },
        ),
        (
            "numpy_nullable",
            {
                "a": "Int64",
                "b": "string",
                "c": "boolean",
                "d": "Float64",
                "e": "datetime64[ns]",
            },
        ),
        (
            [
                "numpy_nullable",
                "pyarrow",
                None,
                "pyarrow",
                "numpy_nullable",
            ],
            {
                "a": "Int64",
                "b": "string[pyarrow]",
                "c": "bool",
                "d": "Float64[pyarrow]",
                "e": "datetime64[ns]",
            },
        ),
    ],
)
def test_schema_to_pandas(
    dtype_backend: DTypeBackend | Sequence[DTypeBackend] | None, expected: dict[str, Any]
) -> None:
    schema = nw_v1.Schema(
        {
            "a": nw_v1.Int64(),
            "b": nw_v1.String(),
            "c": nw_v1.Boolean(),
            "d": nw_v1.Float64(),
            "e": nw_v1.Datetime("ns"),
        }
    )
    assert schema.to_pandas(dtype_backend) == expected


def test_schema_to_pandas_strict_zip() -> None:
    schema = nw_v1.Schema(
        {
            "a": nw_v1.Int64(),
            "b": nw_v1.String(),
            "c": nw_v1.Boolean(),
            "d": nw_v1.Float64(),
            "e": nw_v1.Datetime("ns"),
        }
    )
    dtype_backend: list[DTypeBackend] = ["numpy_nullable", "pyarrow", None]
    tup = (
        "numpy_nullable",
        "pyarrow",
        None,
        "numpy_nullable",
        "pyarrow",
    )
    suggestion = re.escape(f"({tup})")
    with pytest.raises(
        ValueError,
        match=re.compile(
            rf".+3.+but.+schema contains.+5.+field.+Hint.+schema.to_pandas{suggestion}",
            re.DOTALL,
        ),
    ):
        schema.to_pandas(dtype_backend)


def test_schema_to_pandas_invalid() -> None:
    schema = nw_v1.Schema({"a": nw_v1.Int64()})
    msg = "Expected one of {None, 'pyarrow', 'numpy_nullable'}, got: 'cabbage'"
    with pytest.raises(ValueError, match=msg):
        schema.to_pandas("cabbage")  # type: ignore[arg-type]
