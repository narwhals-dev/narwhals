from __future__ import annotations

from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version


def test_to_arrow(request: pytest.FixtureRequest, constructor_eager: Any) -> None:
    if "pandas" in str(constructor_eager) and parse_version(pd.__version__) < (1, 0, 0):
        # pyarrow requires pandas>=1.0.0
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8, 9]}
    df_raw = constructor_eager(data)
    result = nw.from_native(df_raw, eager_only=True).to_arrow()

    expected = pa.table(data)
    assert result == expected
