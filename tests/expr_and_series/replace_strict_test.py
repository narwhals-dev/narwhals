from __future__ import annotations

import pytest
from polars.exceptions import PolarsError

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


def test_replace_strict(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3]}))
    result = df.select(
        nw.col("a").replace_strict(
            {1: "one", 2: "two", 3: "three"}, return_dtype=nw.String
        )
    )
    assert_equal_data(result, {"a": ["one", "two", "three"]})


def test_replace_strict_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}))
    result = df.select(
        df["a"].replace_strict({1: "one", 2: "two", 3: "three"}, return_dtype=nw.String)
    )
    assert_equal_data(result, {"a": ["one", "two", "three"]})


def test_replace_with_default(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor({"a": [1, 2, 3]}))
    if "polars_lazy" in str(constructor):
        with pytest.raises((ValueError, PolarsError)):
            df.lazy().select(
                nw.col("a").replace_strict({1: 3, 3: 4}, return_dtype=nw.Int64)
            ).collect()
    else:
        with pytest.raises((ValueError, PolarsError)):
            df.select(nw.col("a").replace_strict({1: 3, 3: 4}, return_dtype=nw.Int64))
