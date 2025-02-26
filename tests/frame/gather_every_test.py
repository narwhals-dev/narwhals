from __future__ import annotations

from contextlib import nullcontext

import pytest

import narwhals as nw_main
import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data

data = {"a": list(range(10))}


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("offset", [1, 2, 3])
def test_gather_every(
    constructor: Constructor, n: int, offset: int, request: pytest.FixtureRequest
) -> None:
    if "pyspark" in str(constructor) or "duckdb" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df_v1 = nw.from_native(constructor(data))
    result = df_v1.gather_every(n=n, offset=offset)
    expected = {"a": data["a"][offset::n]}
    assert_equal_data(result, expected)

    # Test deprecation for LazyFrame in main namespace
    df_main = nw_main.from_native(constructor(data))

    context = (
        pytest.deprecated_call()
        if isinstance(df_main, nw_main.LazyFrame)
        else nullcontext()
    )

    with context:
        df_main.gather_every(n=n, offset=offset)
