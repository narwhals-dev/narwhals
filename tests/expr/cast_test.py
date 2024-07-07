from typing import Any

import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version


def test_cast(constructor_with_pyarrow: Any, request: Any) -> None:
    if "table" in str(constructor_with_pyarrow) and parse_version(
        pa.__version__
    ) <= parse_version("12.0.0"):  # pragma: no cover
        request.applymarker(pytest.mark.xfail)
    data = {
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
        "n": [True],
        "o": ["a"],
        "p": [1],
    }
    schema = {
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
    df = nw.from_native(constructor_with_pyarrow(data), eager_only=True).select(
        nw.col(key).cast(value) for key, value in schema.items()
    )
    result = df.select(
        nw.col("a").cast(nw.Int32),
        nw.col("b").cast(nw.Int16),
        nw.col("c").cast(nw.Int8),
        nw.col("d").cast(nw.Int64),
        nw.col("e").cast(nw.UInt32),
        nw.col("f").cast(nw.UInt16),
        nw.col("g").cast(nw.UInt8),
        nw.col("h").cast(nw.UInt64),
        nw.col("i").cast(nw.Float32),
        nw.col("j").cast(nw.Float64),
        nw.col("k").cast(nw.String),
        nw.col("l").cast(nw.Datetime),
        nw.col("m").cast(nw.Int8),
        nw.col("n").cast(nw.Int8),
        nw.col("o").cast(nw.String),
        nw.col("p").cast(nw.Duration),
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
    assert result.schema == expected
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
