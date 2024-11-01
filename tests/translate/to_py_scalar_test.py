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
        (np.int64(1), 1),
        (1.0, 1.0),
        (None, None),
        ("a", "a"),
        (True, True),
        (b"a", b"a"),
        (datetime(2021, 1, 1), datetime(2021, 1, 1)),
        (timedelta(days=1), timedelta(days=1)),
    ],
)
def test_to_py_scalar(
    constructor_eager: ConstructorEager,
    input_value: Any,
    expected: Any,
    request: pytest.FixtureRequest,
) -> None:
    if isinstance(input_value, bytes) and "cudf" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)
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


@pytest.mark.parametrize(
    "input_value",
    [np.array([1, 2]), [1, 2, 3], {"a": [1, 2, 3]}],
)
def test_to_py_scalar_value_error(input_value: Any) -> None:
    with pytest.raises(ValueError, match="Expected object convertible to a scalar"):
        nw.to_py_scalar(input_value)


def test_to_py_scalar_value_error_cudf() -> None:
    if cudf := get_cudf():  # pragma: no cover
        df = nw.from_native(cudf.DataFrame({"a": [1, 2, 3]}))

        with pytest.raises(ValueError, match="Expected object convertible to a scalar"):
            nw.to_py_scalar(df["a"])
