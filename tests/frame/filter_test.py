from __future__ import annotations

from contextlib import nullcontext as does_not_raise

import pytest

import narwhals.stable.v1 as nw
from narwhals.exceptions import ShapeError
from tests.utils import Constructor
from tests.utils import assert_equal_data


def test_filter(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))
    result = df.filter(nw.col("a") > 1)
    expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}
    assert_equal_data(result, expected)


def test_filter_with_boolean_list(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))
    context = (
        pytest.raises(TypeError, match="not supported")
        if isinstance(df, nw.LazyFrame)
        else does_not_raise()
    )
    with context:
        result = df.filter([False, True, True])
        expected = {"a": [3, 2], "b": [4, 6], "z": [8.0, 9.0]}
        assert_equal_data(result, expected)


def test_filter_raise_on_agg_predicate(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))

    context = (
        pytest.raises(
            ShapeError,
            match="Predicate result has length 1, which is incompatible with DataFrame of length 3",
        )
        if any(x in str(constructor) for x in ("pandas", "pyarrow", "modin"))
        else does_not_raise()
        if "polars" in str(constructor)
        else pytest.raises(Exception)  # type: ignore[arg-type] # noqa: PT011
    )
    with context:
        df.filter(nw.col("a").max() > 2).lazy().collect()
