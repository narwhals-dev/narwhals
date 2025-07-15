from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import narwhals as nw
from tests.utils import PANDAS_VERSION, is_windows

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


def test_to_numpy(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if (
        "pandas_constructor" in str(constructor_eager)
        or "modin_constructor" in str(constructor_eager)
        or "cudf_constructor" in str(constructor_eager)
    ):
        request.applymarker(pytest.mark.xfail)

    data = [1, 2, None]
    s = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"].cast(
        nw.Int64
    )
    assert s.to_numpy().dtype == "float64"
    assert s.shape == (3,)

    assert_array_equal(s.to_numpy(), np.array(data, dtype=float))


def test_to_numpy_tz_aware(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if (
        ("pandas_pyarrow" in str(constructor_eager) and PANDAS_VERSION < (2, 2))
        or ("modin_pyarrow" in str(constructor_eager) and PANDAS_VERSION < (2, 2))
        or ("pyarrow" in str(constructor_eager) and is_windows())
    ):
        request.applymarker(pytest.mark.xfail)
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(
        constructor_eager({"a": [datetime(2020, 1, 1), datetime(2020, 1, 2)]}),
        eager_only=True,
    )
    df = df.select(nw.col("a").dt.replace_time_zone("Asia/Kathmandu"))
    result = df["a"].to_numpy()
    # for some reason, NumPy uses 'M' for datetimes
    assert result.dtype.kind == "M"
    assert (
        result
        == np.array(
            ["2019-12-31T18:15:00.000000", "2020-01-01T18:15:00.000000"],
            dtype=result.dtype,
        )
    ).all()
