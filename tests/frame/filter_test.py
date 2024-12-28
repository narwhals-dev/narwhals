from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data


def test_filter(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))
    result = df.filter(nw.col("a") > 1)
    expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}
    assert_equal_data(result, expected)


def test_filter_with_boolean_list(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))

    context = (
        pytest.raises(
            NotImplementedError,
            match="`LazyFrame.filter` is not supported for .* with boolean masks.",
        )
        if any(x in str(constructor) for x in ('dask', 'duckdb'))
        else does_not_raise()
    )

    with context:
        result = df.filter([False, True, True])
        expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}
        assert_equal_data(result, expected)
