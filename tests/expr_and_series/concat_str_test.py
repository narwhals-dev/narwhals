from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import compare_dicts

data = {
    "a": [1, 2, 3],
    "b": ["dogs", "cats", None],
    "c": ["play", "swim", "walk"],
}


@pytest.mark.parametrize(
    ("ignore_nulls", "expected"),
    [
        (True, ["2 dogs play", "4 cats swim", "6 walk"]),
        (False, ["2 dogs play", "4 cats swim", None]),
    ],
)
def test_concat_str(
    constructor: Constructor, *, ignore_nulls: bool, expected: list[str]
) -> None:
    df = nw.from_native(constructor(data))
    result = (
        df.select(
            "a",
            nw.concat_str(
                [
                    nw.col("a") * 2,
                    nw.col("b"),
                    nw.col("c"),
                ],
                separator=" ",
                ignore_nulls=ignore_nulls,  # default behavior is False
            ).alias("full_sentence"),
        )
        .sort("a")
        .select("full_sentence")
    )
    compare_dicts(result, {"full_sentence": expected})
    result = (
        df.select(
            "a",
            nw.concat_str(
                nw.col("a") * 2,
                nw.col("b"),
                nw.col("c"),
                separator=" ",
                ignore_nulls=ignore_nulls,  # default behavior is False
            ).alias("full_sentence"),
        )
        .sort("a")
        .select("full_sentence")
    )
    compare_dicts(result, {"full_sentence": expected})
