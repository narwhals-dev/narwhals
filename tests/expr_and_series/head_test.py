from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import compare_dicts


@pytest.mark.parametrize("n", [2, -1])
def test_head_expr(
    constructor: Constructor, n: int, request: pytest.FixtureRequest
) -> None:
    if ("dask" in str(constructor) or "polars" in str(constructor)) and n < 0:
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor({"a": [1, 2, 3]}))
    result = df.select(nw.col("a").head(n))
    expected = {"a": [1, 2]}
    compare_dicts(result, expected)


@pytest.mark.parametrize("n", [2, -1])
def test_head_series(constructor_eager: ConstructorEager, n: int) -> None:
    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)
    result = {"a": df["a"].head(n)}
    expected = {"a": [1, 2]}
    compare_dicts(result, expected)
