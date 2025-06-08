from __future__ import annotations

from typing import Callable

import pyarrow as pa
import pytest

import narwhals as nw
from tests.utils import POLARS_VERSION, Constructor, assert_equal_data

data = {"a": [1, 2, 3], "b": ["dogs", "cats", None], "c": ["play", "swim", "walk"]}


@pytest.mark.parametrize(
    ("ignore_nulls", "expected"),
    [
        (True, ["2 dogs play", "4 cats swim", "6 walk"]),
        (False, ["2 dogs play", "4 cats swim", None]),
    ],
)
def test_concat_str(
    constructor: Constructor,
    *,
    ignore_nulls: bool,
    expected: list[str],
    request: pytest.FixtureRequest,
) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 0, 0):
        # nth only available after 1.0
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor(data))
    result = (
        df.select(
            "a",
            nw.concat_str(
                [nw.col("a") * 2, "b", nw.col("c")],
                separator=" ",
                ignore_nulls=ignore_nulls,  # default behavior is False
            ).alias("full_sentence"),
        )
        .sort("a")
        .select("full_sentence")
    )
    assert_equal_data(result, {"full_sentence": expected})
    result = (
        df.select(
            nw.col("a").alias("a_original"),
            nw.concat_str(
                nw.nth(0) * 2,
                nw.col("b"),
                nw.col("c"),
                separator=" ",
                ignore_nulls=ignore_nulls,  # default behavior is False
            ),
        )
        .sort("a_original")
        .select("a")
    )
    assert_equal_data(result, {"a": expected})


def test_concat_str_with_lit(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": ["cat", "dog", "pig"]}))
    result = df.with_columns(b=nw.concat_str("a", nw.lit("ab")))
    expected = {"a": ["cat", "dog", "pig"], "b": ["catab", "dogab", "pigab"]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("input_schema", "input_values", "expected_function"),
    [
        (
            [("store", pa.large_string()), ("item", pa.large_string())],
            ["a", "b"],
            pa.types.is_large_string,
        ),
        (
            [("store", pa.large_string()), ("item", pa.int32())],
            [0, 1],
            pa.types.is_large_string,
        ),
        ([("store", pa.string()), ("item", pa.int32())], [0, 1], pa.types.is_string),
        ([("store", pa.string()), ("item", pa.string())], ["a", "b"], pa.types.is_string),
    ],
)
def test_pyarrow_string_type(
    input_schema: list[tuple[str, pa.DataType]],
    input_values: list[object],
    expected_function: Callable[[pa.DataType], bool],
) -> None:
    df = pa.table(
        {"store": ["foo", "bar"], "item": input_values}, schema=pa.schema(input_schema)
    )
    result = (
        nw.from_native(df)
        .with_columns(store_item=nw.concat_str("store", "item", separator="-"))
        .to_native()
        .schema
    )
    assert expected_function(result.field("store_item").type)
