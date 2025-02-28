from __future__ import annotations

import re
from datetime import date
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import TYPE_CHECKING
from typing import Any

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import narwhals as nw_main
import narwhals.stable.v1 as nw
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
    df = nw.from_native(
        constructor({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8.0, 9.0]})
    )
    result = df.schema
    expected = {"a": nw.Int64, "b": nw.Int64, "z": nw.Float64}

    result = df.schema
    assert result == expected
    result = df.lazy().collect().schema
    assert result == expected


def test_collect_schema(constructor: Constructor) -> None:
    df = nw.from_native(
        constructor({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8.0, 9.0]})
    )
    expected = {"a": nw.Int64, "b": nw.Int64, "z": nw.Float64}

    result = df.collect_schema()
    assert result == expected
    result = df.lazy().collect().collect_schema()
    assert result == expected


def test_schema_comparison() -> None:
    assert {"a": nw.String()} != {"a": nw.Int32()}
    assert {"a": nw.Int32()} == {"a": nw.Int32()}


def test_object() -> None:
    class Foo: ...

    df = pd.DataFrame({"a": [Foo()]}).astype(object)
    result = nw.from_native(df).schema
    assert result["a"] == nw.Object


def test_string_disguised_as_object() -> None:
    df = pd.DataFrame({"a": ["foo", "bar"]}).astype(object)
    result = nw.from_native(df).schema
    assert result["a"] == nw.String


def test_actual_object(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if any(x in str(constructor_eager) for x in ("pyarrow_table", "cudf")):
        request.applymarker(pytest.mark.xfail)

    class Foo: ...

    data = {"a": [Foo()]}
    df = nw.from_native(constructor_eager(data))
    result = df.schema
    assert result == {"a": nw.Object}


@pytest.mark.skipif(PANDAS_VERSION < (2, 0, 0), reason="too old")
def test_dtypes() -> None:
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
    df_from_pl = nw.from_native(df_pl, eager_only=True)
    expected = {
        "a": nw.Int64,
        "b": nw.Int32,
        "c": nw.Int16,
        "d": nw.Int8,
        "e": nw.UInt64,
        "f": nw.UInt32,
        "g": nw.UInt16,
        "h": nw.UInt8,
        "i": nw.Float64,
        "j": nw.Float32,
        "k": nw.String,
        "l": nw.Datetime,
        "m": nw.Boolean,
        "n": nw.Date,
        "o": nw.Datetime,
        "p": nw.Categorical,
        "q": nw.Duration,
        "r": nw.Enum,
        "s": nw.List,
        "u": nw.Struct,
    }

    assert df_from_pl.schema == df_from_pl.collect_schema()
    assert df_from_pl.schema == expected
    assert {name: df_from_pl[name].dtype for name in df_from_pl.columns} == expected

    # pandas/pyarrow only have categorical, not enum
    expected["r"] = nw.Categorical

    df_from_pd = nw.from_native(
        df_pl.to_pandas(use_pyarrow_extension_array=True), eager_only=True
    )

    assert df_from_pd.schema == df_from_pd.collect_schema() == expected
    assert {name: df_from_pd[name].dtype for name in df_from_pd.columns} == expected

    df_from_pa = nw.from_native(df_pl.to_arrow(), eager_only=True)

    assert df_from_pa.schema == df_from_pa.collect_schema() == expected
    assert {name: df_from_pa[name].dtype for name in df_from_pa.columns} == expected


def test_unknown_dtype() -> None:
    df = pd.DataFrame({"a": pd.period_range("2000", periods=3, freq="M")})
    assert nw.from_native(df).schema == {"a": nw.Unknown}


def test_hash() -> None:
    assert nw.Int64() in {nw.Int64, nw.Int32}


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("names", ["a", "b", "c"]),
        ("dtypes", [nw.Int64(), nw.Float32(), nw.String()]),
        ("len", 3),
    ],
)
def test_schema_object(method: str, expected: Any) -> None:
    data = {"a": nw.Int64(), "b": nw.Float32(), "c": nw.String()}
    schema = nw.Schema(data)
    assert getattr(schema, method)() == expected


def test_validate_not_duplicated_columns_pandas_like() -> None:
    df = pd.DataFrame([[1, 2], [4, 5]], columns=["a", "a"])
    with pytest.raises(
        ValueError, match="Expected unique column names, got:\n- 'a' 2 times"
    ):
        nw.from_native(df, eager_only=True)


def test_validate_not_duplicated_columns_arrow() -> None:
    table = pa.Table.from_arrays([pa.array([1, 2]), pa.array([4, 5])], names=["a", "a"])
    with pytest.raises(
        ValueError, match="Expected unique column names, got:\n- 'a' 2 times"
    ):
        nw.from_native(table, eager_only=True)


def test_validate_not_duplicated_columns_duckdb() -> None:
    duckdb = pytest.importorskip("duckdb")
    rel = duckdb.sql("SELECT 1 AS a, 2 AS a")
    with pytest.raises(
        ValueError, match="Expected unique column names, got:\n- 'a' 2 times"
    ):
        nw.from_native(rel, eager_only=False).lazy().collect()


@pytest.mark.skipif(
    PANDAS_VERSION < (2, 2, 0),
    reason="too old for pyarrow types",
)
def test_nested_dtypes() -> None:
    duckdb = pytest.importorskip("duckdb")
    df_pd = pl.DataFrame(
        {"a": [[1, 2]], "b": [[1, 2]], "c": [{"a": 1}]},
        schema_overrides={"b": pl.Array(pl.Int64, 2)},
    ).to_pandas(use_pyarrow_extension_array=True)
    nwdf: nw.DataFrame[Any] | nw.LazyFrame[Any] = nw.from_native(df_pd)
    assert nwdf.schema == {
        "a": nw.List(nw.Int64),
        "b": nw.Array(nw.Int64, 2),
        "c": nw.Struct({"a": nw.Int64}),
    }
    df_pl = pl.DataFrame(
        {"a": [[1, 2]], "b": [[1, 2]], "c": [{"a": 1}]},
        schema_overrides={"b": pl.Array(pl.Int64, 2)},
    )
    nwdf = nw.from_native(df_pl)
    assert nwdf.schema == {
        "a": nw.List(nw.Int64),
        "b": nw.Array(nw.Int64, 2),
        "c": nw.Struct({"a": nw.Int64}),
    }

    df_pa = pl.DataFrame(
        {"a": [[1, 2]], "b": [[1, 2]], "c": [{"a": 1, "b": "x", "c": 1.1}]},
        schema_overrides={"b": pl.Array(pl.Int64, 2)},
    ).to_arrow()
    nwdf = nw.from_native(df_pa)
    assert nwdf.schema == {
        "a": nw.List(nw.Int64),
        "b": nw.Array(nw.Int64, 2),
        "c": nw.Struct({"a": nw.Int64, "b": nw.String, "c": nw.Float64}),
    }
    rel = duckdb.sql("select * from df_pa")
    nwdf = nw.from_native(rel)
    assert nwdf.schema == {
        "a": nw.List(nw.Int64),
        "b": nw.Array(nw.Int64, 2),
        "c": nw.Struct({"a": nw.Int64, "b": nw.String, "c": nw.Float64}),
    }


def test_nested_dtypes_ibis(request: pytest.FixtureRequest) -> None:  # pragma: no cover
    ibis = pytest.importorskip("ibis")
    if PANDAS_VERSION < (1, 1):
        request.applymarker(pytest.mark.xfail)
    df = pl.DataFrame(
        {"a": [[1, 2]], "b": [[1, 2]], "c": [{"a": 1}]},
        schema_overrides={"b": pl.Array(pl.Int64, 2)},
    )
    tbl = ibis.memtable(df[["a", "c"]])
    nwdf = nw.from_native(tbl)
    assert nwdf.schema == {"a": nw.List(nw.Int64), "c": nw.Struct({"a": nw.Int64})}


@pytest.mark.skipif(
    PANDAS_VERSION < (2, 2, 0),
    reason="too old for pyarrow types",
)
def test_nested_dtypes_dask() -> None:
    pytest.importorskip("dask")
    import dask.dataframe as dd

    df = dd.from_pandas(
        pl.DataFrame(
            {"a": [[1, 2]], "b": [[1, 2]], "c": [{"a": 1}]},
            schema_overrides={"b": pl.Array(pl.Int64, 2)},
        ).to_pandas(use_pyarrow_extension_array=True)
    )
    nwdf = nw.from_native(df)
    assert nwdf.schema == {
        "a": nw.List(nw.Int64),
        "b": nw.Array(nw.Int64, 2),
        "c": nw.Struct({"a": nw.Int64}),
    }


def test_all_nulls_pandas() -> None:
    assert (
        nw_main.from_native(pd.Series([None] * 3, dtype="object"), series_only=True).dtype
        == nw_main.String
    )
    assert (
        nw.from_native(pd.Series([None] * 3, dtype="object"), series_only=True).dtype
        == nw.Object
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
    schema = nw.Schema(
        {
            "a": nw.Int64(),
            "b": nw.String(),
            "c": nw.Boolean(),
            "d": nw.Float64(),
            "e": nw.Datetime("ns"),
        }
    )
    assert schema.to_pandas(dtype_backend) == expected


def test_schema_to_pandas_strict_zip() -> None:
    schema = nw.Schema(
        {
            "a": nw.Int64(),
            "b": nw.String(),
            "c": nw.Boolean(),
            "d": nw.Float64(),
            "e": nw.Datetime("ns"),
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
    schema = nw.Schema({"a": nw.Int64()})
    msg = "Expected one of {None, 'pyarrow', 'numpy_nullable'}, got: 'cabbage'"
    with pytest.raises(ValueError, match=msg):
        schema.to_pandas("cabbage")  # type: ignore[arg-type]
