from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import POLARS_VERSION, ConstructorEager, assert_equal_data


def test_ew_mean(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if any(x in str(constructor_eager) for x in ("pyarrow_table_", "modin")):
        request.applymarker(pytest.mark.xfail)
    if "polars" in str(constructor_eager) and POLARS_VERSION < (1, 10):
        pytest.skip()

    data = {
        "time": [
            "2021-01-01",
            "2021-01-02",
            "2021-01-03",
            "2021-01-04",
            "2021-01-05",
            "2021-01-06",
            "2021-01-07",
            "2021-01-08",
            "2021-01-09",
            "2021-01-10",
        ],
        "close": [
            29331.69,
            32178.33,
            33000.05,
            31988.71,
            33949.53,
            36769.36,
            39432.28,
            40582.81,
            40088.22,
            38150.02,
        ],
    }

    df = nw.from_native(constructor_eager(data))

    expr = (
        nw.col("close")
        .ewm_mean(alpha=0.03)
        .over(order_by="time")
        .round(2)
        .alias("EWMA($\\alpha=.03$)"),
    )

    result = df.select(expr)
    expected = {
        "EWMA($\\alpha=.03$)": [
            29331.69,
            30776.68,
            31540.49,
            31657.72,
            32144.42,
            32975.11,
            33983.95,
            34899.37,
            35548.6,
            35845.82,
        ]
    }
    assert_equal_data(result, expected)
