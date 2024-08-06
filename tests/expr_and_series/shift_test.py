from typing import Any

import pyarrow as pa

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {
    "i": [0, 1, 2, 3, 4],
    "a": [0, 1, 2, 3, 4],
    "b": [1, 2, 3, 5, 3],
    "c": [5, 4, 3, 2, 1],
}


def test_shift(constructor: Any) -> None:
    df = nw.from_native(constructor(data))
    result = df.with_columns(nw.col("a", "b", "c").shift(2)).filter(nw.col("i") > 1)
    expected = {
        "i": [2, 3, 4],
        "a": [0, 1, 2],
        "b": [1, 2, 3],
        "c": [5, 4, 3],
    }
    compare_dicts(result, expected)


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
