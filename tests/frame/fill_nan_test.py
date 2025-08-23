from __future__ import annotations

from datetime import datetime

import narwhals as nw
from tests.utils import Constructor, assert_equal_data


def test_is_nan(constructor: Constructor) -> None:
    df = nw.from_native(
        constructor(
            {
                "a": [1.1, 2.0, float("nan")],
                "b": [3.1, 4.0, None],
                "c": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
            }
        )
    )
    result = df.fill_nan(None)
    expected = {
        "a": [1.1, 2.0, None],
        "b": [3.1, 4.0, None],
        "c": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
    }
    assert_equal_data(result, expected)
    assert result.lazy().collect()["a"].null_count() == 1
    result = df.fill_nan(3.0)
    if any(x in str(constructor) for x in ("pandas", "dask", "cudf", "modin")):
        # pandas doesn't distinguish nan vs null
        expected = {
            "a": [1.1, 2.0, 3.0],
            "b": [3.1, 4.0, 3.0],
            "c": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        }
        assert int(result.lazy().collect()["b"].null_count()) == 0
    else:
        expected = {
            "a": [1.1, 2.0, 3.0],
            "b": [3.1, 4.0, None],
            "c": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        }
        assert result.lazy().collect()["b"].null_count() == 1
    assert_equal_data(result, expected)
    assert result.lazy().collect()["a"].null_count() == 0
