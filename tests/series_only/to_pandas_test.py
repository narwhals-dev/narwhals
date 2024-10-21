from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

import narwhals.stable.v1 as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager

data = [1, 3, 2]


def test_convert(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    pandas_version: tuple[int, ...],
) -> None:
    if pandas_version < (2, 0, 0):
        request.applymarker(pytest.mark.skipif(reason="too old for pyarrow"))
    if any(
        cname in str(constructor_eager)
        for cname in ("pandas_nullable", "pandas_pyarrow", "modin")
    ):
        request.applymarker(pytest.mark.xfail)

    series = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"].alias(
        "a"
    )

    result = series.to_pandas()
    assert_series_equal(result, pd.Series([1, 3, 2], name="a"))
