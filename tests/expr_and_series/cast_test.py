from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, cast

import pandas as pd
import pytest

import narwhals as nw
from tests.utils import (
    PANDAS_VERSION,
    PYARROW_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
    is_windows,
)

if TYPE_CHECKING:
    from narwhals.typing import NativeLazyFrame

DATA = {
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
    "l": [datetime(1970, 1, 1)],
    "m": [True],
    "n": [True],
    "o": ["a"],
    "p": [1],
}
SCHEMA = {
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
    "n": nw.Boolean,
    "o": nw.Categorical,
    "p": nw.Int64,
}

SPARK_LIKE_INCOMPATIBLE_COLUMNS = {"e", "f", "g", "h", "o", "p"}
DUCKDB_INCOMPATIBLE_COLUMNS = {"o"}
IBIS_INCOMPATIBLE_COLUMNS = {"o"}


@pytest.mark.filterwarnings("ignore:casting period[M] values to int64:FutureWarning")
def test_cast(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "pyarrow_table_constructor" in str(constructor) and PYARROW_VERSION <= (
        15,
    ):  # pragma: no cover
        pytest.skip()
    if "modin_constructor" in str(constructor):
        # TODO(unassigned): in modin, we end up with `'<U0'` dtype
        request.applymarker(pytest.mark.xfail)

    if "pyspark" in str(constructor):
        incompatible_columns = SPARK_LIKE_INCOMPATIBLE_COLUMNS  # pragma: no cover
    elif "duckdb" in str(constructor):
        incompatible_columns = DUCKDB_INCOMPATIBLE_COLUMNS  # pragma: no cover
    elif "ibis" in str(constructor):
        incompatible_columns = IBIS_INCOMPATIBLE_COLUMNS  # pragma: no cover
    else:
        incompatible_columns = set()

    data = {c: v for c, v in DATA.items() if c not in incompatible_columns}
    schema = {c: t for c, t in SCHEMA.items() if c not in incompatible_columns}

    df = nw.from_native(constructor(data)).select(
        nw.col(col_).cast(dtype) for col_, dtype in schema.items()
    )

    cast_map = {
        "a": nw.Int32,
        "b": nw.Int16,
        "c": nw.Int8,
        "d": nw.Int64,
        "e": nw.UInt32,
        "f": nw.UInt16,
        "g": nw.UInt8,
        "h": nw.UInt64,
        "i": nw.Float32,
        "j": nw.Float64,
        "k": nw.String,
        "l": nw.Datetime,
        "m": nw.Int8,
        "n": nw.Int8,
        "o": nw.String,
        "p": nw.Duration,
    }
    cast_map = {c: t for c, t in cast_map.items() if c not in incompatible_columns}

    result = df.select(*[nw.col(col_).cast(dtype) for col_, dtype in cast_map.items()])
    assert dict(result.collect_schema()) == cast_map


def test_cast_series(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "pyarrow_table_constructor" in str(constructor_eager) and PYARROW_VERSION <= (
        15,
    ):  # pragma: no cover
        request.applymarker(pytest.mark.xfail)
    if "modin_constructor" in str(constructor_eager):
        # TODO(unassigned): in modin, we end up with `'<U0'` dtype
        request.applymarker(pytest.mark.xfail)
    df = (
        nw.from_native(constructor_eager(DATA))
        .select(nw.col(key).cast(value) for key, value in SCHEMA.items())
        .lazy()
        .collect()
    )

    expected = {
        "a": nw.Int32,
        "b": nw.Int16,
        "c": nw.Int8,
        "d": nw.Int64,
        "e": nw.UInt32,
        "f": nw.UInt16,
        "g": nw.UInt8,
        "h": nw.UInt64,
        "i": nw.Float32,
        "j": nw.Float64,
        "k": nw.String,
        "l": nw.Datetime,
        "m": nw.Int8,
        "n": nw.Int8,
        "o": nw.String,
        "p": nw.Duration,
    }
    result = df.select(
        df["a"].cast(nw.Int32),
        df["b"].cast(nw.Int16),
        df["c"].cast(nw.Int8),
        df["d"].cast(nw.Int64),
        df["e"].cast(nw.UInt32),
        df["f"].cast(nw.UInt16),
        df["g"].cast(nw.UInt8),
        df["h"].cast(nw.UInt64),
        df["i"].cast(nw.Float32),
        df["j"].cast(nw.Float64),
        df["k"].cast(nw.String),
        df["l"].cast(nw.Datetime),
        df["m"].cast(nw.Int8),
        df["n"].cast(nw.Int8),
        df["o"].cast(nw.String),
        df["p"].cast(nw.Duration),
    )
    assert result.schema == expected


def test_cast_string() -> None:
    s_pd = pd.Series([1, 2]).convert_dtypes()
    s = nw.from_native(s_pd, series_only=True)
    s = s.cast(nw.String)
    result = nw.to_native(s)
    assert str(result.dtype) in {"string", "object", "dtype('O')"}


def test_cast_raises_for_unknown_dtype(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    if "pyarrow_table" in str(constructor) and PYARROW_VERSION < (15,):
        # Unsupported cast from string to dictionary using function cast_dictionary
        request.applymarker(pytest.mark.xfail)

    if "pyspark" in str(constructor):
        incompatible_columns = SPARK_LIKE_INCOMPATIBLE_COLUMNS  # pragma: no cover
    elif "ibis" in str(constructor):
        incompatible_columns = IBIS_INCOMPATIBLE_COLUMNS  # pragma: no cover
    else:
        incompatible_columns = set()

    data = {k: v for k, v in DATA.items() if k not in incompatible_columns}
    schema = {k: v for k, v in SCHEMA.items() if k not in incompatible_columns}

    df = nw.from_native(constructor(data)).select(
        nw.col(key).cast(value) for key, value in schema.items()
    )

    class Banana:
        pass

    with pytest.raises(TypeError, match="Expected Narwhals dtype"):
        df.select(nw.col("a").cast(Banana))  # type: ignore[arg-type]


def test_cast_datetime_tz_aware(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if (
        "dask" in str(constructor)
        or "duckdb" in str(constructor)
        or "cudf" in str(constructor)  # https://github.com/rapidsai/cudf/issues/16973
        or ("pyarrow_table" in str(constructor) and is_windows())
        or "pyspark" in str(constructor)
        or "ibis" in str(constructor)
    ):
        request.applymarker(pytest.mark.xfail)

    data = {
        "date": [
            datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
            for i in range(3)
        ]
    }
    expected = {
        "date": ["2024-01-01 01:00:00", "2024-01-02 01:00:00", "2024-01-03 01:00:00"]
    }

    df = nw.from_native(constructor(data))
    result = df.select(
        nw.col("date")
        .cast(nw.Datetime("ms", time_zone="Europe/Rome"))
        .cast(nw.String())
        .str.slice(offset=0, length=19)
    )
    assert_equal_data(result, expected)


def test_cast_datetime_utc(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if (
        "dask" in str(constructor)
        # https://github.com/eakmanrq/sqlframe/issues/406
        or "sqlframe" in str(constructor)
        or ("pyarrow_table" in str(constructor) and is_windows())
    ):
        request.applymarker(pytest.mark.xfail)

    data = {
        "date": [
            datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
            for i in range(3)
        ]
    }
    expected = {
        "date": ["2024-01-01 00:00:00", "2024-01-02 00:00:00", "2024-01-03 00:00:00"]
    }

    df = nw.from_native(constructor(data))
    result = df.select(
        nw.col("date")
        .cast(nw.Datetime("us", time_zone="UTC"))
        .cast(nw.String())
        .str.slice(offset=0, length=19)
    )
    assert_equal_data(result, expected)


def test_cast_struct(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(
        backend in str(constructor) for backend in ("dask", "modin", "cudf", "sqlframe")
    ):
        request.applymarker(pytest.mark.xfail)

    if "pandas" in str(constructor) and PANDAS_VERSION < (2, 2):
        pytest.skip()

    data = {
        "a": [{"movie ": "Cars", "rating": 4.5}, {"movie ": "Toy Story", "rating": 4.9}]
    }

    native_df = constructor(data)

    # NOTE: This branch needs to be rewritten to **not depend** on private `SparkLikeLazyFrame` properties
    if "spark" in str(constructor):  # pragma: no cover
        # Special handling for pyspark as it natively maps the input to
        # a column of type MAP<STRING, STRING>
        native_ldf = cast("NativeLazyFrame", native_df)
        _tmp_nw_compliant_frame = nw.from_native(native_ldf)._compliant_frame
        F = _tmp_nw_compliant_frame._F  # type: ignore[attr-defined] # noqa: N806
        T = _tmp_nw_compliant_frame._native_dtypes  # type: ignore[attr-defined] # noqa: N806

        native_ldf = native_ldf.withColumn(  # type: ignore[attr-defined]
            "a",
            F.struct(
                F.col("a.movie ").cast(T.StringType()).alias("movie "),
                F.col("a.rating").cast(T.DoubleType()).alias("rating"),
            ),
        )
        assert nw.from_native(native_ldf).schema == nw.Schema(
            {
                "a": nw.Struct(
                    [nw.Field("movie ", nw.String()), nw.Field("rating", nw.Float64())]
                )
            }
        )
        native_df = native_ldf

    dtype = nw.Struct([nw.Field("movie ", nw.String()), nw.Field("rating", nw.Float32())])
    result = nw.from_native(native_df).select(nw.col("a").cast(dtype)).lazy().collect()
    assert result.schema == {"a": dtype}


def test_raise_if_polars_dtype(constructor: Constructor) -> None:
    pytest.importorskip("polars")
    import polars as pl

    for dtype in [pl.String, pl.String()]:
        df = nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
        with pytest.raises(TypeError, match="Expected Narwhals dtype, got:"):
            df.select(nw.col("a").cast(dtype))  # type: ignore[arg-type]


def test_cast_time(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if "pandas" in str(constructor) and PANDAS_VERSION < (2, 2):
        pytest.skip()

    if any(
        backend in str(constructor) for backend in ("dask", "pyspark", "modin", "cudf")
    ):
        request.applymarker(pytest.mark.xfail)

    data = {"a": [time(12, 0, 0), time(12, 0, 5)]}
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").cast(nw.Time()))
    assert result.collect_schema() == {"a": nw.Time()}


def test_cast_binary(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if "pandas" in str(constructor) and PANDAS_VERSION < (2, 2):
        pytest.skip()

    if any(backend in str(constructor) for backend in ("cudf", "dask", "modin")):
        request.applymarker(pytest.mark.xfail)

    data = {"a": ["test1", "test2"]}
    df = nw.from_native(constructor(data))
    result = df.select(
        "a",
        b=nw.col("a").cast(nw.Binary()),
        c=nw.col("a").cast(nw.Binary()).cast(nw.String()),
    )
    assert result.collect_schema() == {
        "a": nw.String(),
        "b": nw.Binary(),
        "c": nw.String(),
    }
    assert_equal_data(result.select("c"), {"c": data["a"]})


def test_cast_typing_invalid() -> None:
    """**IMPORTANT**!

    **Please don't parametrize these tests.**

    They're checking we warn early when a cast will fail at runtime.
    """
    a = nw.col("a")

    with pytest.raises(TypeError):
        a.cast(nw.Field)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        a.cast(nw.Field("a", nw.Array))  # type: ignore[arg-type]

    pytest.importorskip("polars")
    import polars as pl

    df = nw.from_native(pl.DataFrame({"a": [1]}))

    # NOTE: If you're seeing this fail because they raise a more consistent error,
    # feel free to update the types used
    # See (https://github.com/narwhals-dev/narwhals/pull/2654#discussion_r2142263770)

    with pytest.raises(AttributeError):
        df.select(a.cast(nw.Struct))  # type: ignore[arg-type]

    with pytest.raises(AttributeError):
        df.select(a.cast(nw.List))  # type: ignore[arg-type]

    with pytest.raises(AttributeError):
        df.select(a.cast(nw.Array))  # type: ignore[arg-type]

    with pytest.raises(ValueError):  # noqa: PT011
        df.select(a.cast(nw.Enum))  # type: ignore[arg-type]

    with pytest.raises(AttributeError):
        df.select(a.cast(nw.Struct([nw.Field])))  # type: ignore[list-item]

    with pytest.raises((ValueError, AttributeError)):
        df.select(a.cast(nw.Struct({"a": nw.Int16, "b": nw.Enum})))  # type: ignore[dict-item]

    with pytest.raises(AttributeError):
        df.select(a.cast(nw.List(nw.Struct)))  # type: ignore[arg-type]

    with pytest.raises(AttributeError):
        df.select(a.cast(nw.Array(nw.List, 2)))  # type: ignore[arg-type]
