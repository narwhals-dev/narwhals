from contextlib import nullcontext as does_not_raise
from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_filter(constructor: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))
    result = df.filter(nw.col("a") > 1)
    expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}
    compare_dicts(result, expected)


@pytest.mark.filterwarnings("ignore:If `index_col` is not specified for `to_spark`")
def test_filter_with_boolean_list(constructor: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))

    context = (
        pytest.raises(
            NotImplementedError,
            match="`LazyFrame.filter` is not supported for Dask backend with boolean masks.",
        )
        if "dask" in str(constructor) or "pyspark" in str(constructor)
        else does_not_raise()
    )

    with context:
        result = df.filter([False, True, True])
        expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}
        compare_dicts(result, expected)
