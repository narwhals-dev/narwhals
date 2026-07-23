from __future__ import annotations

from math import isnan
from typing import Any

import pytest

import narwhals as nw
from tests.utils import (
    POLARS_VERSION,
    ConstructorEager,
    assert_equal_data,
    assert_equal_series,
)

polars_lt_v1 = POLARS_VERSION < (1, 0, 0)
pl_skip_reason = "replace_strict only available after 1.0"


@pytest.mark.parametrize(
    ("values", "expected_n_unique"),
    [
        ([], 0),
        ([*"abcabc"], 3),
        ([1, 2, 3, 2], 3),
        ([1.1, 2.2, 3.3, 2.2], 3),
        ([*"abc", None], 3),
        ([*"aaabbbccc", None], 3),
    ],
)
def test_factorize_invariants(
    values: list[Any], expected_n_unique: int, constructor_eager: ConstructorEager
) -> None:
    if "polars" in str(constructor_eager) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    has_null = any(x is None for x in values)

    df_native = constructor_eager({"a": values})
    df = nw.from_native(df_native)
    codes, uniqs = df["a"].factorize()

    reconstructed_values = {"a": [uniqs[i] if i >= 0 else None for i in codes]}
    assert_equal_data(df, reconstructed_values)
    assert uniqs.dtype == df["a"].dtype
    assert len(uniqs) == expected_n_unique

    # codes should be integer, preserve length, and only contain -1 in the presence of nulls
    assert codes.dtype.is_integer()
    assert len(codes) == len(values)
    assert (codes >= -1).all()
    assert (codes == -1).any() == has_null

    # Null values should always be dropped out from the unique returned values
    assert not (uniqs.is_null().any())


@pytest.mark.parametrize(
    ("values", "expected_uniqs", "expected_codes"),
    [
        ([], [], []),
        ([*"abc"], [*"abc"], [0, 1, 2]),
        ([*"abcabc"], [*"abc"], [0, 1, 2, 0, 1, 2]),
        ([*"aaabbbccc"], [*"abc"], [0, 0, 0, 1, 1, 1, 2, 2, 2]),
        ([*"abcabc", None], [*"abc"], [0, 1, 2, 0, 1, 2, -1]),
    ],
)
def test_factorize_sort(
    values: list[Any],
    expected_uniqs: list[Any],
    expected_codes: list[int],
    constructor_eager: ConstructorEager,
) -> None:
    if "polars" in str(constructor_eager) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    df_native = constructor_eager({"a": values})
    df = nw.from_native(df_native)
    codes, uniqs = df["a"].factorize(sort=True)

    assert_equal_series(uniqs, expected_uniqs, name="a")
    assert_equal_series(codes, expected_codes, name="a")


@pytest.mark.parametrize(
    "values",
    [
        [1.1, 2.2, 1.1, float("nan")],
        [1.1, 2.2, 1.1, float("nan"), float("nan")],
        [1.1, 2.2, 1.1, None, float("nan")],
    ],
)
def test_factorize_nan_semantics(
    values: list[float], constructor_eager: ConstructorEager
) -> None:
    if "polars" in str(constructor_eager) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    is_pandas_backend = any(x in str(constructor_eager) for x in ("pandas", "modin"))

    df_native = constructor_eager({"a": values})
    df = nw.from_native(df_native)
    codes, uniqs = df["a"].factorize()

    reconstructed_values = {"a": [uniqs[i] if i >= 0 else None for i in codes]}
    assert_equal_data(df, reconstructed_values)

    if is_pandas_backend:
        # pandas treats NaN as missing, so NaN is not retained as a unique value.
        assert len(uniqs) == 2
        assert (codes == -1).any()
        assert not uniqs.is_null().any()
    else:
        # Other backends treat NaN as a value, not as null.
        assert len(uniqs) == 3

        # The NaN should round-trip through codes -> uniques.
        nan_index = (
            i
            for i, value in enumerate(values)
            if isinstance(value, float) and isnan(value)
        )
        nan_codes = (codes[nan_i] for nan_i in nan_index)
        assert all(isnan(uniqs[nan_c]) for nan_c in nan_codes)
