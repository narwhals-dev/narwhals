from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest

import narwhals as nw
from tests.utils import PANDAS_VERSION, is_windows

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


def test_to_numpy(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.1, 8.0, 9.0]}
    df_raw = constructor_eager(data)
    result = nw.from_native(df_raw, eager_only=True).to_numpy()
    expected = np.array([[1, 3, 2], [4, 4, 6], [7.1, 8.0, 9.0]]).T
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == "float64"


def test_to_numpy_tz_aware(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if ("pandas_pyarrow" in str(constructor_eager) and PANDAS_VERSION < (2, 2)) or (
        "pyarrow" in str(constructor_eager) and is_windows()
    ):
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(
        constructor_eager({"a": [datetime(2020, 1, 1), datetime(2020, 1, 2)]}),
        eager_only=True,
    )
    df = df.select(nw.col("a").dt.replace_time_zone("Asia/Kathmandu"))
    result = df.to_numpy()
    # for some reason, NumPy uses 'M' for datetimes
    assert result.dtype.kind == "M"
    assert (
        result
        == np.array(
            [["2019-12-31T18:15:00.000000"], ["2020-01-01T18:15:00.000000"]],
            dtype=result.dtype,
        )
    ).all()
