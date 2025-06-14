from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import narwhals as nw
from narwhals.stable.v1.dependencies import get_cudf


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        (1, 1),
        (pa.scalar(1), 1),
        (np.int64(1), 1),
        (Decimal("1.1"), Decimal("1.1")),
        (1.0, 1.0),
        (None, None),
        ("a", "a"),
        (True, True),
        (b"a", b"a"),
        (datetime(2021, 1, 1), datetime(2021, 1, 1)),
        (timedelta(days=1), timedelta(days=1)),
        (date(1980, 1, 1), date(1980, 1, 1)),
        (time(9, 45), time(9, 45)),
        (pd.Timestamp("2020-01-01"), datetime(2020, 1, 1)),
        (pd.Timedelta(days=3), timedelta(days=3)),
        (np.datetime64("2020-01-01", "s"), datetime(2020, 1, 1)),
        (np.datetime64("2020-01-01", "ms"), datetime(2020, 1, 1)),
        (np.datetime64("2020-01-01", "us"), datetime(2020, 1, 1)),
        (np.datetime64("2020-01-01", "ns"), datetime(2020, 1, 1)),
    ],
)
def test_to_py_scalar(input_value: Any, expected: Any) -> None:
    output = nw.to_py_scalar(input_value)
    if expected == 1:
        assert not isinstance(output, np.generic)
    assert output == expected


def test_na_to_py_scalar() -> None:
    assert nw.to_py_scalar(pd.NA) is None


@pytest.mark.parametrize("input_value", [np.array([1, 2]), [1, 2, 3], {"a": [1, 2, 3]}])
def test_to_py_scalar_value_error(input_value: Any) -> None:
    with pytest.raises(ValueError, match="Expected object convertible to a scalar"):
        nw.to_py_scalar(input_value)


def test_to_py_scalar_value_error_cudf() -> None:
    if cudf := get_cudf():  # pragma: no cover
        df = nw.from_native(cudf.DataFrame({"a": [1, 2, 3]}))

        with pytest.raises(ValueError, match="Expected object convertible to a scalar"):
            nw.to_py_scalar(df["a"])
