from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import pandas as pd
import pytest

import narwhals.stable.v1 as nw
from narwhals.dependencies import get_cudf

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        (1, 1),
        (datetime(2021, 1, 1), datetime(2021, 1, 1)),
        (timedelta(days=1), timedelta(days=1)),
    ],
)
def test_to_py_scalar(
    constructor_eager: ConstructorEager, input_value: Any, expected: Any
) -> None:
    df = nw.from_native(constructor_eager({"a": [input_value]}))
    output = nw.to_py_scalar(df["a"].item(0))
    if expected == 1 and constructor_eager.__name__.startswith("pandas"):
        assert not isinstance(output, np.int64)
    elif isinstance(expected, datetime) and constructor_eager.__name__.startswith(
        "pandas"
    ):
        assert not isinstance(output, pd.Timestamp)
    elif isinstance(expected, timedelta) and constructor_eager.__name__.startswith(
        "pandas"
    ):
        assert not isinstance(output, pd.Timedelta)
    assert output == expected


def test_to_py_scalar_arrays_series() -> None:
    if cudf := get_cudf():  # pragma: no cover
        df = nw.from_native(cudf.DataFrame({"a": [1, 2, 3]}))
        cudf_series = nw.to_native(nw.to_py_scalar(df["a"]))
        assert isinstance(cudf_series, cudf.Series)

    array = np.array([1, 2, 3])
    assert isinstance(nw.to_py_scalar(array), np.ndarray)
