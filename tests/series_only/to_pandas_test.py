from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

import narwhals as nw
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
    from tests.utils import ConstructorEager

data = [1, 3, 2]


@pytest.mark.skipif(PANDAS_VERSION < (2, 0, 0), reason="too old for pyarrow")
def test_convert(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if any(
        cname in str(constructor_eager)
        for cname in ("pandas_nullable", "pandas_pyarrow", "modin_pyarrow")
    ):
        request.applymarker(pytest.mark.xfail)

    series = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"].alias(
        "a"
    )

    result = series.to_pandas()
    assert_series_equal(result, pd.Series([1, 3, 2], name="a"))
