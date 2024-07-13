from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = [1, 2, 3]


def test_to_frame(constructor_series: Any) -> None:
    if "pyarrow_series" in str(constructor_series):
        pytest.xfail()

    df = nw.from_native(constructor_series(data), series_only=True).alias("").to_frame()
    compare_dicts(df, {"": [1, 2, 3]})
