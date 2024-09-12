from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


@pytest.mark.parametrize("expr1", ["a", nw.col("a")])
@pytest.mark.parametrize("expr2", ["b", nw.col("b")])
def test_anyh(
    constructor: Any, expr1: Any, expr2: Any, request: pytest.FixtureRequest
) -> None:
    if "pyspark" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    data = {
        "a": [False, False, True],
        "b": [False, True, True],
    }
    df = nw.from_native(constructor(data))
    result = df.select(any=nw.any_horizontal(expr1, expr2))

    expected = {"any": [False, True, True]}
    compare_dicts(result, expected)
