from __future__ import annotations

import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe


def test_len_chars() -> None:
    data = {"a": ["foo", "foobar", "Café", "345", "東京"]}
    expected = {"a": [3, 6, 4, 3, 2]}
    result = dataframe(data).select(nwp.col("a").str.len_chars())
    assert_equal_data(result, expected)
