from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor, assert_equal_data


def test_format(constructor: Constructor) -> None:
    df = nw.from_native(
        constructor(
            {
                "name": ["bob", "alice", "dodo"],
                "surname": ["builder", "wonderlander", "extinct"],
            }
        )
    )
    result = df.select(fmt=nw.format("hello {} {} wassup", "name", nw.col("surname")))
    expected = {
        "fmt": [
            "hello bob builder wassup",
            "hello alice wonderlander wassup",
            "hello dodo extinct wassup",
        ]
    }
    assert_equal_data(result, expected)
    result = df.select(fmt=nw.format("{} {} wassup", "name", nw.col("surname")))
    expected = {
        "fmt": ["bob builder wassup", "alice wonderlander wassup", "dodo extinct wassup"]
    }
    assert_equal_data(result, expected)


def test_format_invalid() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    df = nw.from_native(
        pd.DataFrame(
            {
                "name": ["bob", "alice", "dodo"],
                "surname": ["builder", "wonderlander", "extinct"],
            }
        )
    )
    with pytest.raises(ValueError, match="Expected 2 arguments, got 1"):
        df.select(fmt=nw.format("hello {} {} wassup", "name"))
