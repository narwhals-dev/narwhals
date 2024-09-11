from typing import Any

import pytest

import narwhals.stable.v1 as nw

data = {"a": [0, 0, 2, -1], "b": [1, 3, 2, None]}


@pytest.mark.parametrize(
    ("descending", "nulls_last", "expected"),
    [
        (True, True, {"b": [3, 2, 1, None]}),
        (True, False, {"b": [None, 3, 2, 1]}),
        (False, True, {"b": [1, 2, 3, None]}),
        (False, False, {"b": [None, 1, 2, 3]}),
    ],
)
def test_sort_single_expr(
    constructor: Any, descending: Any, nulls_last: Any, expected: Any
) -> None:
    df = nw.from_native(constructor(data)).lazy()
    result = nw.to_native(
        df.select(
            nw.col("b").sort(descending=descending, nulls_last=nulls_last),
        ).collect()
    )

    expected_df = nw.to_native(nw.from_native(constructor(expected)).lazy().collect())
    result = nw.maybe_align_index(result, expected_df)
    assert result.equals(expected_df)


@pytest.mark.parametrize(
    ("descending", "nulls_last", "expected"),
    [
        (True, True, {"a": [0, 0, 2, -1], "b": [3, 2, 1, None]}),
        (True, False, {"a": [0, 0, 2, -1], "b": [None, 3, 2, 1]}),
        (False, True, {"a": [0, 0, 2, -1], "b": [1, 2, 3, None]}),
        (False, False, {"a": [0, 0, 2, -1], "b": [None, 1, 2, 3]}),
    ],
)
def test_sort_multiple_expr(
    constructor: Any, descending: Any, nulls_last: Any, expected: Any, request: Any
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data)).lazy()
    result = nw.to_native(
        df.select(
            "a",
            nw.col("b").sort(descending=descending, nulls_last=nulls_last),
        ).collect()
    )

    expected_df = nw.to_native(nw.from_native(constructor(expected)).lazy().collect())
    assert result.equals(expected_df)


@pytest.mark.parametrize(
    ("descending", "nulls_last", "expected"),
    [
        (True, True, [3, 2, 1, None]),
        (True, False, [None, 3, 2, 1]),
        (False, True, [1, 2, 3, None]),
        (False, False, [None, 1, 2, 3]),
    ],
)
def test_sort_series(
    constructor_eager: Any, descending: Any, nulls_last: Any, expected: Any
) -> None:
    series = nw.from_native(constructor_eager(data), eager_only=True)["b"]
    result = series.sort(descending=descending, nulls_last=nulls_last)
    assert (
        result == nw.from_native(constructor_eager({"a": expected}), eager_only=True)["a"]
    )
