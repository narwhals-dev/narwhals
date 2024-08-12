from __future__ import annotations

from typing import Any

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version

data = [1, 3, 2]


@pytest.mark.skipif(
    parse_version(pd.__version__) < parse_version("2.0.0"), reason="too old for pyarrow"
)
def test_convert(request: Any, constructor_eager: Any) -> None:
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
