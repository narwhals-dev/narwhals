from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
    from tests.utils import Constructor, ConstructorEager


def test_cast_253(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "pyarrow_table" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    df_raw = constructor_eager({"a": [1]})
    result = nw.from_native(df_raw, eager_only=True).select(
        nw.col("a").cast(nw.String) + "hi"
    )["a"][0]
    assert result == "1hi"


def test_cast_date_datetime_polars() -> None:
    pytest.importorskip("polars")
    import polars as pl
    from polars.testing import assert_frame_equal

    # polars: date to datetime
    dfpl = pl.DataFrame({"a": [date(2020, 1, 1), date(2020, 1, 2)]})
    df = nw.from_native(dfpl)
    df = df.select(nw.col("a").cast(nw.Datetime))
    result = nw.to_native(df)
    expected = pl.DataFrame({"a": [datetime(2020, 1, 1), datetime(2020, 1, 2)]})
    assert_frame_equal(result, expected)

    # polars: datetime to date
    dfpl = pl.DataFrame({"a": [datetime(2020, 1, 1), datetime(2020, 1, 2)]})
    df = nw.from_native(dfpl)
    df = df.select(nw.col("a").cast(nw.Date))
    result = nw.to_native(df)
    expected = pl.DataFrame({"a": [date(2020, 1, 1), date(2020, 1, 2)]})
    assert_frame_equal(result, expected)
    assert df.schema == {"a": nw.Date}


def test_cast_date_datetime_pyarrow() -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    # polars: date to datetime
    dfpa = pa.table({"a": [date(2020, 1, 1), date(2020, 1, 2)]})
    df = nw.from_native(dfpa)
    df = df.select(nw.col("a").cast(nw.Datetime))
    result = nw.to_native(df)
    expected = pa.table({"a": [datetime(2020, 1, 1), datetime(2020, 1, 2)]})
    assert result == expected

    # pyarrow: datetime to date
    dfpa = pa.table({"a": [datetime(2020, 1, 1), datetime(2020, 1, 2)]})
    df = nw.from_native(dfpa)
    df = df.select(nw.col("a").cast(nw.Date))
    result = nw.to_native(df)
    expected = pa.table({"a": [date(2020, 1, 1), date(2020, 1, 2)]})
    assert result == expected


@pytest.mark.skipif(PANDAS_VERSION < (2, 0, 0), reason="pyarrow dtype not available")
def test_cast_date_datetime_pandas() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    # pandas: pyarrow date to datetime
    dfpd = pd.DataFrame({"a": [date(2020, 1, 1), date(2020, 1, 2)]}).astype(
        {"a": "date32[pyarrow]"}
    )
    df = nw.from_native(dfpd)
    df = df.select(nw.col("a").cast(nw.Datetime))
    result = nw.to_native(df)
    expected = pd.DataFrame({"a": [datetime(2020, 1, 1), datetime(2020, 1, 2)]}).astype(
        {"a": "timestamp[us][pyarrow]"}
    )
    pd.testing.assert_frame_equal(result, expected)

    # pandas: pyarrow datetime to date
    dfpd = pd.DataFrame({"a": [datetime(2020, 1, 1), datetime(2020, 1, 2)]}).astype(
        {"a": "timestamp[us][pyarrow]"}
    )
    df = nw.from_native(dfpd)
    df = df.select(nw.col("a").cast(nw.Date))
    result = nw.to_native(df)
    expected = pd.DataFrame({"a": [date(2020, 1, 1), date(2020, 1, 2)]}).astype(
        {"a": "date32[pyarrow]"}
    )
    pd.testing.assert_frame_equal(result, expected)
    assert df.schema == {"a": nw.Date}


@pytest.mark.filterwarnings("ignore: casting period")
def test_unknown_to_int() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = pd.DataFrame({"a": pd.period_range("2000", periods=3, freq="min")})
    assert nw.from_native(df).select(nw.col("a").cast(nw.Int64)).schema == {"a": nw.Int64}


def test_cast_to_enum_vmain(
    request: pytest.FixtureRequest, constructor: Constructor
) -> None:
    # Backends that do not (yet) support Enum dtype
    if any(
        backend in str(constructor)
        for backend in ("pyarrow_table", "sqlframe", "pyspark", "ibis")
    ):
        request.applymarker(pytest.mark.xfail)

    df_nw = nw.from_native(constructor({"a": ["a", "b"]}))
    col_a = nw.col("a")

    with pytest.raises(
        ValueError, match="Can not cast / initialize Enum without categories present"
    ):
        df_nw.select(col_a.cast(nw.Enum))  # type: ignore[arg-type]

    df_nw = df_nw.select(col_a.cast(nw.Enum(["a", "b"])))
    assert df_nw.collect_schema() == {"a": nw.Enum(["a", "b"])}
