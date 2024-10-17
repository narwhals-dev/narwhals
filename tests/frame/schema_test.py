from datetime import date
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any

import duckdb
import pandas as pd
import polars as pl
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version
from tests.utils import Constructor
from tests.utils import ConstructorEager

data = {
    "a": [datetime(2020, 1, 1)],
    "b": [datetime(2020, 1, 1, tzinfo=timezone.utc)],
}


@pytest.mark.filterwarnings("ignore:Determining|Resolving.*")
def test_schema(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8, 9]}))
    result = df.schema
    expected = {"a": nw.Int64, "b": nw.Int64, "z": nw.Float64}

    result = df.schema
    assert result == expected
    result = df.lazy().collect().schema
    assert result == expected


def test_collect_schema(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8, 9]}))
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
    if any(x in str(constructor_eager) for x in ("modin", "pyarrow_table", "cudf")):
        request.applymarker(pytest.mark.xfail)

    class Foo: ...

    data = {"a": [Foo()]}
    df = nw.from_native(constructor_eager(data))
    result = df.schema
    assert result == {"a": nw.Object}


@pytest.mark.skipif(
    parse_version(pd.__version__) < parse_version("2.0.0"), reason="too old"
)
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


@pytest.mark.skipif(
    parse_version(pd.__version__) < (2,),
    reason="Before 2.0, pandas would raise on `drop_duplicates`",
)
def test_from_non_hashable_column_name() -> None:
    # This is technically super-illegal
    # BUT, it shows up in a scikit-learn test, so...
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["pizza", ["a", "b"]])

    df = nw.from_native(df, eager_only=True)
    assert df.columns == ["pizza", ["a", "b"]]
    assert df["pizza"].dtype == nw.Int64


@pytest.mark.skipif(
    parse_version(pd.__version__) < parse_version("2.2.0"),
    reason="too old for pyarrow types",
)
def test_nested_dtypes() -> None:
    df = pl.DataFrame(
        {"a": [[1, 2]], "b": [[1, 2]], "c": [{"a": 1}]},
        schema_overrides={"b": pl.Array(pl.Int64, 2)},
    ).to_pandas(use_pyarrow_extension_array=True)
    nwdf = nw.from_native(df)
    assert nwdf.schema == {
        "a": nw.List(nw.Int64),
        "b": nw.Array(nw.Int64, 2),
        "c": nw.Struct({"a": nw.Int64}),
    }
    df = pl.DataFrame(
        {"a": [[1, 2]], "b": [[1, 2]], "c": [{"a": 1}]},
        schema_overrides={"b": pl.Array(pl.Int64, 2)},
    )
    nwdf = nw.from_native(df)
    assert nwdf.schema == {
        "a": nw.List(nw.Int64),
        "b": nw.Array(nw.Int64, 2),
        "c": nw.Struct({"a": nw.Int64}),
    }

    df = pl.DataFrame(
        {"a": [[1, 2]], "b": [[1, 2]], "c": [{"a": 1, "b": "x", "c": 1.1}]},
        schema_overrides={"b": pl.Array(pl.Int64, 2)},
    ).to_arrow()
    nwdf = nw.from_native(df)
    assert nwdf.schema == {
        "a": nw.List(nw.Int64),
        "b": nw.Array(nw.Int64, 2),
        "c": nw.Struct({"a": nw.Int64, "b": nw.String, "c": nw.Float64}),
    }
    df = duckdb.sql("select * from df")
    nwdf = nw.from_native(df)
    assert nwdf.schema == {
        "a": nw.List(nw.Int64),
        "b": nw.Array(nw.Int64, 2),
        "c": nw.Struct({"a": nw.Int64, "b": nw.String, "c": nw.Float64}),
    }


def test_nested_dtypes_ibis() -> None:  # pragma: no cover
    ibis = pytest.importorskip("ibis")
    df = pl.DataFrame(
        {"a": [[1, 2]], "b": [[1, 2]], "c": [{"a": 1}]},
        schema_overrides={"b": pl.Array(pl.Int64, 2)},
    )
    tbl = ibis.memtable(df[["a", "c"]])
    nwdf = nw.from_native(tbl)
    assert nwdf.schema == {"a": nw.List(nw.Int64), "c": nw.Struct({"a": nw.Int64})}


@pytest.mark.skipif(
    parse_version(pd.__version__) < parse_version("2.2.0"),
    reason="too old for pyarrow types",
)
def test_nested_dtypes_dask() -> None:
    pytest.importorskip("dask")
    pytest.importorskip("dask_expr", exc_type=ImportError)
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
