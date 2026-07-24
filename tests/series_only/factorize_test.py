from __future__ import annotations

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
    ("values", "null_as_value", "expected_n_unique"),
    [
        ([], False, 0),
        ([], True, 0),
        ([*"abcabc"], False, 3),
        ([*"abcabc"], True, 3),
        ([1, 2, 3, 2], False, 3),
        ([1, 2, 3, 2], True, 3),
        ([1.1, 2.2, 3.3, 2.2], False, 3),
        ([1.1, 2.2, 3.3, 2.2], True, 3),
        ([*"abc", None], False, 3),
        ([*"abc", None], True, 4),
        ([*"aaabbbccc", None], False, 3),
        ([*"aaabbbccc", None], True, 4),
    ],
)
def test_factorize_invariants(
    constructor_eager: ConstructorEager,
    *,
    values: list[Any],
    null_as_value: bool,
    expected_n_unique: int,
) -> None:
    if "polars" in str(constructor_eager) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    has_null = any(x is None for x in values)

    df_native = constructor_eager({"a": values})
    df = nw.from_native(df_native)
    codes, uniqs = df["a"].factorize(null_as_value=null_as_value)

    reconstructed_values = {"a": [uniqs[i] if i >= 0 else None for i in codes]}
    assert_equal_data(df, reconstructed_values)
    assert uniqs.dtype == df["a"].dtype
    assert len(uniqs) == expected_n_unique

    # codes should be integer, preserve length, and
    #   only contain -1 in the presence of nulls in input & `null_as_value=False`
    assert codes.dtype.is_integer()
    assert len(codes) == len(values)

    min_code = 0 if null_as_value else -1
    assert (codes >= min_code).all()
    assert (codes == -1).any() == (has_null and not null_as_value)


@pytest.mark.parametrize(
    ("values", "null_as_value", "expected_uniqs", "expected_codes"),
    [
        ([], False, [], []),
        ([None], False, [], [-1]),
        ([*"abc"], False, [*"abc"], [0, 1, 2]),
        ([*"abcabc"], False, [*"abc"], [0, 1, 2, 0, 1, 2]),
        ([*"abcabc", None], False, [*"abc"], [0, 1, 2, 0, 1, 2, -1]),
        ([*"aaabbbccc"], False, [*"abc"], [0, 0, 0, 1, 1, 1, 2, 2, 2]),
        ([10, 11, 12], False, [10, 11, 12], [0, 1, 2]),
        ([10, 11, 12, 10, 11, 12], False, [10, 11, 12], [0, 1, 2, 0, 1, 2]),
        ([10, 10, 11, 11, 12, 12], False, [10, 11, 12], [0, 0, 1, 1, 2, 2]),
        ([10, 11, 12, None], False, [10, 11, 12], [0, 1, 2, -1]),
        ([], True, [], []),
        ([None], True, [None], [0]),
        ([*"abc"], True, [*"abc"], [0, 1, 2]),
        ([*"abcabc"], True, [*"abc"], [0, 1, 2, 0, 1, 2]),
        ([*"abcabc", None], True, [*"abc", None], [0, 1, 2, 0, 1, 2, 3]),
        ([*"aaabbbccc"], True, [*"abc"], [0, 0, 0, 1, 1, 1, 2, 2, 2]),
        ([10, 11, 12], True, [10, 11, 12], [0, 1, 2]),
        ([10, 11, 12, 10, 11, 12], True, [10, 11, 12], [0, 1, 2, 0, 1, 2]),
        ([10, 10, 11, 11, 12, 12], True, [10, 11, 12], [0, 0, 1, 1, 2, 2]),
        ([10, 11, 12, None], True, [10, 11, 12, None], [0, 1, 2, 3]),
    ],
)
def test_factorize_sort(
    constructor_eager: ConstructorEager,
    *,
    values: list[Any],
    null_as_value: bool,
    expected_uniqs: list[Any],
    expected_codes: list[int],
) -> None:
    if "polars" in str(constructor_eager) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    df_native = constructor_eager({"a": values})
    df = nw.from_native(df_native)
    codes, uniqs = df["a"].factorize(null_as_value=null_as_value, sort=True)

    assert_equal_series(uniqs, expected_uniqs, name="a")
    assert_equal_series(codes, expected_codes, name="a")


@pytest.mark.parametrize(
    ("values", "null_as_value", "expected_unique", "expected_unique_pandas"),
    [
        ([1.1, 2.2, 1.1, float("nan")], False, [1.1, 2.2, float("nan")], [1.1, 2.2]),
        (
            [1.1, 2.2, 1.1, float("nan"), float("nan")],
            False,
            [1.1, 2.2, float("nan")],
            [1.1, 2.2],
        ),
        (
            [1.1, 2.2, 1.1, None, float("nan")],
            False,
            [1.1, 2.2, float("nan")],
            [1.1, 2.2],
        ),
        ([1.1, 2.2, 1.1, float("nan")], True, [1.1, 2.2, float("nan")], [1.1, 2.2, None]),
        (
            [1.1, 2.2, 1.1, float("nan"), float("nan")],
            True,
            [1.1, 2.2, float("nan")],
            [1.1, 2.2, None],
        ),
        (
            [1.1, 2.2, 1.1, None, float("nan")],
            True,
            [1.1, 2.2, float("nan"), None],
            [1.1, 2.2, None],
        ),
    ],
)
def test_factorize_nan_semantics(
    constructor_eager: ConstructorEager,
    *,
    values: list[float | None],
    null_as_value: bool,
    expected_unique: list[Any],
    expected_unique_pandas: list[Any],
) -> None:
    if "polars" in str(constructor_eager) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    is_pandas_backend = any(x in str(constructor_eager) for x in ("pandas", "modin"))
    expected = expected_unique_pandas if is_pandas_backend else expected_unique

    df_native = constructor_eager({"a": values})
    df = nw.from_native(df_native)
    codes, uniqs = df["a"].factorize(null_as_value=null_as_value, sort=True)

    reconstructed_values = {"a": [uniqs[i] if i >= 0 else None for i in codes]}
    assert_equal_data(df, reconstructed_values)
    assert_equal_series(uniqs, expected, name="a")
