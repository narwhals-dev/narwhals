from __future__ import annotations

import pytest

import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (
            nwp.format("hello {} {} wassup", "name", nwp.col("surname")),
            [
                "hello bob builder wassup",
                "hello alice wonderlander wassup",
                "hello dodo extinct wassup",
            ],
        ),
        (
            nwp.format("{} {} wassup", "name", nwp.col("surname")),
            ["bob builder wassup", "alice wonderlander wassup", "dodo extinct wassup"],
        ),
    ],
)
def test_format(expr: nwp.Expr, expected: list[str]) -> None:
    data = {
        "name": ["bob", "alice", "dodo"],
        "surname": ["builder", "wonderlander", "extinct"],
    }
    result = dataframe(data).select(fmt=expr)
    assert_equal_data(result, {"fmt": expected})


def test_format_invalid() -> None:
    with pytest.raises(ValueError, match="Expected 2 arguments, got 1"):
        nwp.format("hello {} {} wassup", "name")
