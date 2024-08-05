from __future__ import annotations

from typing import Any

import pyarrow as pa

import narwhals as nw
from tests.utils import compare_dicts


def test_shift(constructor_eager: Any) -> None:
    data = {
        "A": [1, 2, None, 4],
        "B": [5, 6, 7, 8],
        "C": [None, None, None, None],
        "D": [9, 10, 11, 12],
    }

    df = nw.from_native(constructor_eager(data), eager_only=True)

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
    df = nw.from_native(tbl, eager_only=True)

    a_type = tbl.schema.field("a").type

    # Shift right by 1
    result_a = df.select(nw.col("a").shift(1))
    # Prepend a null and exclude the last element
    expected_a = pa.concat_arrays(
        [pa.nulls(1, a_type)]
        + [chunk.slice(0, len(chunk) - 1) for chunk in tbl.column(0).chunks]
    )
    expected_a_dict = {"a": expected_a.to_pylist()}

    # Shift left by -1
    result_b = df.select(nw.col("a").shift(-1))
    # Exclude the first element and append a null
    expected_b = pa.concat_arrays(
        [chunk.slice(1, len(chunk) - 1) for chunk in tbl.column(0).chunks]
        + [pa.nulls(1, a_type)]
    )
    expected_b_dict = {"a": expected_b.to_pylist()}

    # when shift is 0
    result_c = df.select(nw.col("a").shift(0))
    expected_c = tbl.column(0)
    expected_c_dict = {
        "a": [item for chunk in expected_c.chunks for item in chunk.to_pylist()]
    }

    compare_dicts(result_a, expected_a_dict)
    compare_dicts(result_b, expected_b_dict)
    compare_dicts(result_c, expected_c_dict)
