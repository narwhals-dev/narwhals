from __future__ import annotations

import re

import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, Constructor, assert_equal_data

data = {
    "OriginCityName": [
        "Columbus, OH",
        "Newark, NJ",
        "Philadelphia, PA",
        "Nashville, TN",
        "Washington, DC",
        "Phoenix, AZ",
    ]
}

expected = {
    "OriginCityName": [
        "Columbus",
        "Newark",
        "Philadelphia",
        "Nashville",
        "Washington",
        "Phoenix",
    ]
}


def test_split_list_get(
    request: pytest.FixtureRequest, nw_frame_constructor: Constructor
) -> None:
    if any(backend in str(nw_frame_constructor) for backend in ("dask",)):
        request.applymarker(pytest.mark.xfail)

    if "pandas" in str(nw_frame_constructor):
        if PANDAS_VERSION < (2, 2):
            pytest.skip()
        pytest.importorskip("pyarrow")
    if str(nw_frame_constructor).startswith("pandas") and "pyarrow" not in str(
        nw_frame_constructor
    ):
        df = nw.from_native(nw_frame_constructor(data))
        msg = re.escape("This operation requires a pyarrow-backed series. ")
        with pytest.raises(TypeError, match=msg):
            df.select(nw.col("OriginCityName").str.split(",").list.get(0))
        return
    df = nw.from_native(nw_frame_constructor(data))
    result = df.select(nw.col("OriginCityName").str.split(",").list.get(0))
    assert_equal_data(result, expected)
