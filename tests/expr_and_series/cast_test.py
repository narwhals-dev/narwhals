from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, cast

import pytest

import narwhals as nw
from tests.utils import (
    PANDAS_VERSION,
    PYARROW_VERSION,
    Constructor,
    ConstructorEager,
    assert_equal_data,
    is_pyarrow_windows_no_tzdata,
    time_unit_compat,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from narwhals._native import NativeSQLFrame
    from narwhals.typing import NonNestedDType

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
SCHEMA: Mapping[str, type[NonNestedDType]] = {
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
MODIN_XFAIL_COLUMNS = {"o", "k"}


@pytest.mark.filterwarnings("ignore:casting period[M] values to int64:FutureWarning")
def test_cast(constructor: Constructor) -> None:
    if "pyarrow_table_constructor" in str(constructor) and PYARROW_VERSION <= (
        15,
    ):  # pragma: no cover
        pytest.skip()

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

    cast_map: Mapping[str, type[NonNestedDType]] = {
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

    result = df.select(
        *[nw.col(col_).cast(dtype) for col_, dtype in cast_map.items()]
    ).collect_schema()

    for (key, ltype), rtype in zip(result.items(), cast_map.values()):
        if "modin_constructor" in str(constructor) and key in MODIN_XFAIL_COLUMNS:
            # TODO(unassigned): in modin we end up with `'<U0'` dtype
            # This block will act similarly to an xfail i.e. if we fix the issue, the
            # assert will fail
            assert ltype != rtype
        else:
            assert ltype == rtype, f"types differ for column {key}: {ltype}!={rtype}"


def test_cast_series(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "pyarrow_table_constructor" in str(constructor_eager) and PYARROW_VERSION <= (
        15,
    ):  # pragma: no cover
        request.applymarker(pytest.mark.xfail)

    df = (
        nw.from_native(constructor_eager(DATA))
        .select(nw.col(key).cast(value) for key, value in SCHEMA.items())
        .lazy()
        .collect()
    )
    cast_map: Mapping[str, type[NonNestedDType]] = {
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
    result = df.select(df[col_].cast(dtype) for col_, dtype in cast_map.items()).schema

    for (key, ltype), rtype in zip(result.items(), cast_map.values()):
        if "modin_constructor" in str(constructor_eager) and key in MODIN_XFAIL_COLUMNS:
            # TODO(unassigned): in modin we end up with `'<U0'` dtype
            # This block will act similarly to an xfail i.e. if we fix the issue, the
            # assert will fail
            assert ltype != rtype
        else:
            assert ltype == rtype, f"types differ for column {key}: {ltype}!={rtype}"


def test_cast_string() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

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
        or "pyspark" in str(constructor)
        or "ibis" in str(constructor)
    ):
        request.applymarker(pytest.mark.xfail)
    request.applymarker(
        pytest.mark.xfail(is_pyarrow_windows_no_tzdata(constructor), reason="no tzdata")
    )

    data = {
        "date": [
            datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
            for i in range(3)
        ]
    }
    expected = {
        "date": ["2024-01-01 01:00:00", "2024-01-02 01:00:00", "2024-01-03 01:00:00"]
    }
    dtype = nw.Datetime(time_unit_compat("ms", request), time_zone="Europe/Rome")
    df = nw.from_native(constructor(data))
    result = df.select(
        nw.col("date").cast(dtype).cast(nw.String()).str.slice(offset=0, length=19)
    )
    assert_equal_data(result, expected)


def test_cast_datetime_utc(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if (
        "dask" in str(constructor)
        # https://github.com/eakmanrq/sqlframe/issues/406
        or "sqlframe" in str(constructor)
    ):
        request.applymarker(pytest.mark.xfail)
    request.applymarker(
        pytest.mark.xfail(is_pyarrow_windows_no_tzdata(constructor), reason="no tzdata")
    )

    data = {
        "date": [
            datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)
            for i in range(3)
        ]
    }
    expected = {
        "date": ["2024-01-01 00:00:00", "2024-01-02 00:00:00", "2024-01-03 00:00:00"]
    }
    dtype = nw.Datetime(time_unit_compat("us", request), time_zone="UTC")
    df = nw.from_native(constructor(data))
    result = df.select(
        nw.col("date").cast(dtype).cast(nw.String()).str.slice(offset=0, length=19)
    )
    assert_equal_data(result, expected)


def test_cast_struct(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(backend in str(constructor) for backend in ("dask", "cudf", "sqlframe")):
        request.applymarker(pytest.mark.xfail)

    if "pandas" in str(constructor):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")

    data = {
        "a": [{"movie ": "Cars", "rating": 4.5}, {"movie ": "Toy Story", "rating": 4.9}]
    }

    native_df = constructor(data)

    # NOTE: This branch needs to be rewritten to **not depend** on private `SparkLikeLazyFrame` properties
    if "spark" in str(constructor):  # pragma: no cover
        # Special handling for pyspark as it natively maps the input to
        # a column of type MAP<STRING, STRING>
        native_ldf = cast("NativeSQLFrame", native_df)
        _tmp_nw_compliant_frame = nw.from_native(native_ldf)._compliant_frame
        F = _tmp_nw_compliant_frame._F  # type: ignore[attr-defined]
        T = _tmp_nw_compliant_frame._native_dtypes  # type: ignore[attr-defined] # noqa: N806

        native_ldf = native_ldf.withColumn(
            "a",
            F.struct(
                F.col("a.movie ").cast(T.StringType()).alias("movie "),
                F.col("a.rating").cast(T.DoubleType()).alias("rating"),
            ),
        )
        assert nw.from_native(native_ldf).collect_schema() == nw.Schema(
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
    if "pandas" in str(constructor):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")

    if any(backend in str(constructor) for backend in ("dask", "pyspark", "cudf")):
        request.applymarker(pytest.mark.xfail)

    data = {"a": [time(12, 0, 0), time(12, 0, 5)]}
    df = nw.from_native(constructor(data))
    result = df.select(nw.col("a").cast(nw.Time()))
    assert result.collect_schema() == {"a": nw.Time()}


def test_cast_binary(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if "pandas" in str(constructor):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")

    if any(backend in str(constructor) for backend in ("cudf", "dask")):
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

    with pytest.raises(TypeError):
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

    with pytest.raises(TypeError):
        df.select(a.cast(nw.List(nw.Struct)))  # type: ignore[arg-type]

    with pytest.raises(AttributeError):
        df.select(a.cast(nw.Array(nw.List, 2)))  # type: ignore[arg-type]


@pytest.mark.skipif(PANDAS_VERSION < (2,), reason="too old for pyarrow")
def test_pandas_pyarrow_dtypes() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import pandas as pd

    s = nw.from_native(
        pd.Series([123, None]).convert_dtypes(dtype_backend="pyarrow"), series_only=True
    ).cast(nw.String)
    result = s.str.len_chars().to_native()
    assert result.dtype == "Int32[pyarrow]"

    s = nw.from_native(
        pd.Series([123, None], dtype="string[pyarrow]"), series_only=True
    ).cast(nw.String)
    result = s.str.len_chars().to_native()
    assert result.dtype == "Int64"

    s = nw.from_native(
        pd.DataFrame({"a": ["foo", "bar"]}, dtype="string[pyarrow]")
    ).select(nw.col("a").cast(nw.String))["a"]
    assert s.to_native().dtype == "string[pyarrow]"


def test_cast_object_pandas() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    s = nw.from_native(pd.DataFrame({"a": [2, 3, None]}, dtype=object))["a"]
    assert s[0] == 2
    assert s.cast(nw.String)[0] == "2"
