from __future__ import annotations

from typing import Any

import pyarrow as pa

import narwhals as nw
from tests.utils import compare_dicts


def test_shift(constructor: Any) -> None:
    data = {
        "A": [1, 2, None, 4],
        "B": [5, 6, 7, 8],
        "C": [None, None, None, None],
        "D": [9, 10, 11, 12],
    }

    df = nw.from_native(constructor(data), eager_only=True)

    result_a = df.select(nw.col("A").shift(1))
    result_b = df.select(nw.col("B").shift(-1))
    result_c = df.select(nw.col("C").shift(1))
    result_d = df.select(nw.col("D").shift(2))

    expected_a = {"A": [float("nan"), 1.0, 2.0, float("nan")]}
    expected_b = {"B": [6.0, 7.0, 8.0, float("nan")]}
    expected_c = {"C": [float("nan"), float("nan"), float("nan"), float("nan")]}
    expected_d = {"D": [float("nan"), float("nan"), 9, 10]}

    compare_dicts(result_a, expected_a)
    compare_dicts(result_b, expected_b)
    compare_dicts(result_c, expected_c)
    compare_dicts(result_d, expected_d)


def test_shift_multi_chunk_pyarrow() -> None:
    tbl = pa.table({"a": [1, 2, 3]})
    tbl = pa.concat_tables([tbl, tbl, tbl])
    tbl = tbl.combine_chunks()
    df = nw.from_native(tbl, eager_only=True)
    result_a = df.select(nw.col("a").shift(1))
    result_b = df.select(nw.col("a").shift(-1))
    expected_a = {"a": [None, 1, 2, 3, 1, 2, 3, 1, 2]}
    expected_b = {"a": [2, 3, 1, 2, 3, 1, 2, 3, None]}

    compare_dicts(result_a, expected_a)
    compare_dicts(result_b, expected_b)
