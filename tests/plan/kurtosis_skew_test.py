from __future__ import annotations

import pytest

from narwhals import _plan as nwp
from tests.plan.utils import assert_equal_data, dataframe


@pytest.mark.parametrize(
    ("data", "expected_kurtosis", "expected_skew"),
    [
        ([], None, None),
        ([1], None, None),
        ([1, 2], -2, 0.0),
        ([0.0, 0.0, 0.0], None, None),
        ([1, 2, 3, 2, 1], -1.153061, 0.343622),
    ],
    ids=range(5),
)
def test_kurtosis_skew_expr(
    data: list[float], expected_kurtosis: float | None, expected_skew: float | None
) -> None:
    df = dataframe({"a": data})
    kurtosis = nwp.col("a").kurtosis()
    skew = nwp.col("a").skew()
    height = len(data)

    assert_equal_data(df.select(kurtosis), {"a": [expected_kurtosis]})
    assert_equal_data(df.select(skew), {"a": [expected_skew]})
    assert_equal_data(df.with_columns(kurtosis), {"a": [expected_kurtosis] * height})
    assert_equal_data(df.with_columns(skew), {"a": [expected_skew] * height})
