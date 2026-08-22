from __future__ import annotations

from typing import Any

import pytest

import narwhals as nw
from tests.utils import Constructor, assert_equal_data, skip_if_no_categorical_ordering


def test_sort(constructor: Constructor) -> None:
    data = {"an tan": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor(data))
    result = df.sort("an tan", "b")
    expected = {"an tan": [1, 2, 3], "b": [4, 6, 4], "z": [7.0, 9.0, 8.0]}
    assert_equal_data(result, expected)
    result = df.sort("an tan", "b", descending=[True, False])
    expected = {"an tan": [3, 2, 1], "b": [4, 6, 4], "z": [8.0, 9.0, 7.0]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("nulls_last", "expected"),
    [
        (True, {"antan desc": [0, 2, 0, -1], "b": [3, 2, 1, None]}),
        (False, {"antan desc": [-1, 0, 2, 0], "b": [None, 3, 2, 1]}),
    ],
)
def test_sort_nulls(
    constructor: Constructor, *, nulls_last: bool, expected: dict[str, float]
) -> None:
    data = {"antan desc": [0, 0, 2, -1], "b": [1, 3, 2, None]}
    df = nw.from_native(constructor(data))
    result = df.sort("b", descending=True, nulls_last=nulls_last)
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("descending", "nulls_last", "expected"),
    [
        (False, True, {"c": ["bird", "cat", "dog", None, None], "n": [4, 3, 1, 2, 5]}),
        (False, False, {"c": [None, None, "bird", "cat", "dog"], "n": [2, 5, 4, 3, 1]}),
        (True, True, {"c": ["dog", "cat", "bird", None, None], "n": [1, 3, 4, 2, 5]}),
        (True, False, {"c": [None, None, "dog", "cat", "bird"], "n": [2, 5, 1, 3, 4]}),
    ],
)
def test_sort_categorical(
    constructor: Constructor,
    *,
    descending: bool,
    nulls_last: bool,
    expected: dict[str, Any],
) -> None:
    # (unordered) categoricals order lexicographically on the value, not on the
    # encoding: https://github.com/narwhals-dev/narwhals/issues/3841
    skip_if_no_categorical_ordering(constructor)

    data = {"c": ["dog", None, "cat", "bird", None], "n": [1, 2, 3, 4, 5]}
    df = nw.from_native(constructor(data)).with_columns(
        nw.col("c").cast(nw.Categorical())
    )
    result = df.sort("c", "n", descending=[descending, False], nulls_last=nulls_last)
    assert_equal_data(result, expected)


def test_sort_categorical_empty(constructor: Constructor) -> None:
    skip_if_no_categorical_ordering(constructor)

    data = {"c": ["dog", "cat"], "n": [1, 2]}
    df = nw.from_native(constructor(data)).with_columns(
        nw.col("c").cast(nw.Categorical())
    )
    result = df.filter(nw.col("n") > 2).sort("c")
    assert_equal_data(result, {"c": [], "n": []})


@pytest.mark.parametrize(
    ("chunks", "ordered", "expected"),
    [
        ([([0, 1, 0], ["a", None])], False, [None, "a", "a"]),
        ([([2, 0, 1], [3, 1, 2])], False, [1, 2, 3]),
        ([([0, 1, 2], ["b", "a", "b"])], False, ["a", "b", "b"]),
        (
            [([0, 1], ["dog", "cat"]), ([0, 1], ["bird", "dog"])],
            False,
            ["bird", "cat", "dog", "dog"],
        ),
        ([([0, 1, 2], ["dog", "cat", "bird"])], True, ["bird", "cat", "dog"]),
    ],
)
def test_sort_dictionary_pyarrow(
    chunks: list[tuple[list[int], list[Any]]], expected: list[Any], *, ordered: bool
) -> None:
    """Dictionary layouts which cannot be expressed through a `constructor`."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    column = pa.chunked_array(
        [
            pa.DictionaryArray.from_arrays(
                pa.array(indices, pa.int32()), pa.array(values)
            )
            for indices, values in chunks
        ]
    )
    if ordered:
        column = column.cast(pa.dictionary(pa.int32(), pa.string(), ordered=True))
    native = pa.Table.from_arrays([column], names=["c"])
    result = nw.from_native(native).sort("c", nulls_last=False)
    assert_equal_data(result, {"c": expected})
    # ordering does not decode the output
    assert result.to_native().schema == native.schema
