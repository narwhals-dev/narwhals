# Test assorted functions which we overwrite in stable.v1
from __future__ import annotations

import narwhals.stable.v1 as nw_v1
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


def test_agg_shorthands(constructor_eager: ConstructorEager) -> None:
    df = nw_v1.from_native(constructor_eager({"a": [1, 2, 3]}))
    result = df.select(
        min=nw_v1.min("a"),
        max=nw_v1.max("a"),
        mean=nw_v1.mean("a"),
        median=nw_v1.median("a"),
        sum=nw_v1.sum("a"),
        sum_h=nw_v1.sum_horizontal("a"),
        min_h=nw_v1.min_horizontal("a"),
        max_h=nw_v1.max_horizontal("a"),
        mean_h=nw_v1.mean_horizontal("a"),
        len=nw_v1.len(),
    )
    expected = {
        "min": [1, 1, 1],
        "max": [3, 3, 3],
        "mean": [2.0, 2.0, 2.0],
        "median": [2.0, 2.0, 2.0],
        "sum": [6, 6, 6],
        "sum_h": [1, 2, 3],
        "min_h": [1, 2, 3],
        "max_h": [1, 2, 3],
        "mean_h": [1, 2, 3],
        "len": [3, 3, 3],
    }
    assert_equal_data(result, expected)
