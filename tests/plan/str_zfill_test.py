from __future__ import annotations

import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe


def test_str_zfill() -> None:
    data = {"a": ["-1", "+1", "1", "12", "123", "99999", "+9999", None]}
    expected = {"a": ["-01", "+01", "001", "012", "123", "99999", "+9999", None]}
    result = dataframe(data).select(nwp.col("a").str.zfill(3))
    assert_equal_data(result, expected)
