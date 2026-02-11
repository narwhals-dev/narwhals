from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, assert_equal_data

data = {"weight": ["89kg", "83", "79kg", "68kg", "78kg", "73", "86kg"]}

expected = {"weight_kg": [89, 38, 79, 68, 78, 33, 86]}


def test_parse_weight(request: pytest.FixtureRequest, constructor: Constructor) -> None:
    if any(x in str(constructor) for x in ("pyspark", "ibis", "duckdb")):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))

    col = nw.col("weight")
    is_kg = col.str.ends_with("kg")
    without_unit = col.str.replace("kg", "").cast(nw.UInt8)
    pounds_to_kg = (without_unit * 0.453592).round(0).cast(nw.UInt8)

    result = df.select(
        nw.when(is_kg).then(without_unit).otherwise(pounds_to_kg).alias("weight_kg")
    )

    assert_equal_data(result, expected)
