from datetime import date

import ibis
import polars as pl
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version


@pytest.mark.skipif(
    parse_version(ibis.__version__) < (6, 0),
    reason="too old, requires interchange protocol",
)
def test_interchange_schema() -> None:
    df_pl = pl.DataFrame(
        {
            "a": [1, 1, 2],
            "b": [4, 5, 6],
            "c": [4, 5, 6],
            "d": [4, 5, 6],
            "e": [4, 5, 6],
            "f": [4, 5, 6],
            "g": [4, 5, 6],
            "h": [4, 5, 6],
            "i": [4, 5, 6],
            "j": [4, 5, 6],
            "k": ["fdafsd", "fdas", "ad"],
            "l": ["fdafsd", "fdas", "ad"],
            "m": [date(2021, 1, 1), date(2021, 1, 1), date(2021, 1, 1)],
            "n": [True, True, False],
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
            "l": pl.Categorical,
            "m": pl.Datetime,
            "n": pl.Boolean,
        },
    )
    tbl = ibis.memtable(df_pl)
    df = nw.from_native(tbl, eager_or_interchange_only=True)
    result = df.schema
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
        "l": nw.String,  # https://github.com/ibis-project/ibis/issues/9570
        "m": nw.Datetime,
        "n": nw.Boolean,
    }
    assert result == expected
    assert df["a"].dtype == nw.Int64


@pytest.mark.skipif(
    parse_version(ibis.__version__) < (6, 0),
    reason="too old, requires interchange protocol",
)
def test_invalid() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    tbl = ibis.memtable(df)
    with pytest.raises(
        NotImplementedError, match="is not supported for metadata-only dataframes"
    ):
        nw.from_native(tbl, eager_or_interchange_only=True).select("a")
    with pytest.raises(TypeError, match="Cannot only use `series_only=True`"):
        nw.from_native(tbl, eager_only=True)


@pytest.mark.skipif(
    parse_version(ibis.__version__) < (6, 0),
    reason="too old, requires interchange protocol",
)
def test_get_level() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    tbl = ibis.memtable(df)
    assert nw.get_level(nw.from_native(tbl, eager_or_interchange_only=True)) == "metadata"
    assert nw.get_level(nw.from_native(df)) == "full"
