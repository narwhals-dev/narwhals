from datetime import date
from datetime import datetime

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import narwhals as nw
from narwhals.utils import parse_version


def test_cast_253() -> None:
    df_polars = pl.DataFrame({"a": [1]})
    result = nw.from_native(df_polars, eager_only=True).select(
        nw.col("a").cast(nw.String) + "hi"
    )["a"][0]
    assert result == "1hi"

    df_pandas = pd.DataFrame({"a": [1]})
    result = nw.from_native(df_pandas, eager_only=True).select(
        nw.col("a").cast(nw.String) + "hi"
    )["a"][0]
    assert result == "1hi"


def test_cast_date_datetime_polars() -> None:
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


@pytest.mark.skipif(
    parse_version(pd.__version__) < parse_version("2.0.0"),
    reason="pyarrow dtype not available",
)
def test_cast_date_datetime_pandas() -> None:
    # pandas: pyarrow date to datetime
    dfpd = pd.DataFrame({"a": [date(2020, 1, 1), date(2020, 1, 2)]}).astype(
        {"a": "date32[pyarrow]"}
    )
    df = nw.from_native(dfpd)
    df = df.select(nw.col("a").cast(nw.Datetime))
    result = nw.to_native(df)
    expected = pd.DataFrame({"a": [datetime(2020, 1, 1), datetime(2020, 1, 2)]}).astype(
        {"a": "timestamp[ns][pyarrow]"}
    )
    pd.testing.assert_frame_equal(result, expected)

    # pandas: pyarrow datetime to date
    dfpd = pd.DataFrame({"a": [datetime(2020, 1, 1), datetime(2020, 1, 2)]}).astype(
        {"a": "timestamp[ns][pyarrow]"}
    )
    df = nw.from_native(dfpd)
    df = df.select(nw.col("a").cast(nw.Date))
    result = nw.to_native(df)
    expected = pd.DataFrame({"a": [date(2020, 1, 1), date(2020, 1, 2)]}).astype(
        {"a": "date32[pyarrow]"}
    )
    pd.testing.assert_frame_equal(result, expected)
    assert df.schema == {"a": nw.Date}


@pytest.mark.skipif(
    parse_version(pd.__version__) < parse_version("2.0.0"),
    reason="pyarrow dtype not available",
)
def test_cast_date_datetime_invalid() -> None:
    # pandas: pyarrow datetime to date
    dfpd = pd.DataFrame({"a": [datetime(2020, 1, 1), datetime(2020, 1, 2)]})
    df = nw.from_native(dfpd)
    with pytest.raises(NotImplementedError, match="pyarrow"):
        df.select(nw.col("a").cast(nw.Date))
