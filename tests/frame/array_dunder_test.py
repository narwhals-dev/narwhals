from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from narwhals.utils import parse_version
from tests.utils import ConstructorEager
from tests.utils import compare_dicts


def test_array_dunder(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "pyarrow_table" in str(constructor_eager) and parse_version(
        pa.__version__
    ) < parse_version("16.0.0"):  # pragma: no cover
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)
    result = df.__array__()
    np.testing.assert_array_equal(result, np.array([[1], [2], [3]], dtype="int64"))


def test_array_dunder_with_dtype(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "pyarrow_table" in str(constructor_eager) and parse_version(
        pa.__version__
    ) < parse_version("16.0.0"):  # pragma: no cover
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)
    result = df.__array__(object)
    np.testing.assert_array_equal(result, np.array([[1], [2], [3]], dtype=object))


def test_array_dunder_with_copy(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    if "pyarrow_table" in str(constructor_eager) and parse_version(pa.__version__) < (
        16,
        0,
        0,
    ):  # pragma: no cover
        request.applymarker(pytest.mark.xfail)
    if "polars" in str(constructor_eager) and parse_version(pl.__version__) < (
        0,
        20,
        28,
    ):  # pragma: no cover
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)
    result = df.__array__(copy=True)
    np.testing.assert_array_equal(result, np.array([[1], [2], [3]], dtype="int64"))
    if "pandas_constructor" in str(constructor_eager) and parse_version(
        pd.__version__
    ) < (3,):
        # If it's pandas, we know that `copy=False` definitely took effect.
        # So, let's check it!
        result = df.__array__(copy=False)
        np.testing.assert_array_equal(result, np.array([[1], [2], [3]], dtype="int64"))
        result[0, 0] = 999
        compare_dicts(df, {"a": [999, 2, 3]})
