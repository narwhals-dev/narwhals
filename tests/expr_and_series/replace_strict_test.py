from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import POLARS_VERSION
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data


@pytest.mark.skipif(
    POLARS_VERSION < (1, 0), reason="replace_strict only available after 1.0"
)
def test_replace_strict(constructor: Constructor, request: pytest.FixtureRequest) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor({"a": [1, 2, 3]}))
    result = df.select(
        nw.col("a").replace_strict(
            [1, 2, 3], ["one", "two", "three"], return_dtype=nw.String
        )
    )
    assert_equal_data(result, {"a": ["one", "two", "three"]})


@pytest.mark.skipif(
    POLARS_VERSION < (1, 0), reason="replace_strict only available after 1.0"
)
def test_replace_strict_series(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}))
    result = df.select(
        df["a"].replace_strict([1, 2, 3], ["one", "two", "three"], return_dtype=nw.String)
    )
    assert_equal_data(result, {"a": ["one", "two", "three"]})


@pytest.mark.skipif(
    POLARS_VERSION < (1, 0), reason="replace_strict only available after 1.0"
)
def test_replace_non_full(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    from polars.exceptions import PolarsError

    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor({"a": [1, 2, 3]}))
    if isinstance(df, nw.LazyFrame):
        with pytest.raises((ValueError, PolarsError)):
            df.select(
                nw.col("a").replace_strict([1, 3], [3, 4], return_dtype=nw.Int64)
            ).collect()
    else:
        with pytest.raises((ValueError, PolarsError)):
            df.select(nw.col("a").replace_strict([1, 3], [3, 4], return_dtype=nw.Int64))
