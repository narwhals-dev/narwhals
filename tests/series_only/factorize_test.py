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

NULL_ARGS = [
    NULL_PRESERVE := {"null_policy": "preserve"},
    NULL_ENCODE := {"null_policy": "encode"},
    NULL_SENTINEL := {"null_policy": "sentinel", "sentinel": -1},
]


@pytest.mark.parametrize(
    ("values", "expected_unique"),
    [
        ([], []),
        ([*"abcabc"], ["a", "b", "c"]),
        ([1, 2, 3, 2], [1, 2, 3]),
        ([1.1, 2.2, 3.3, 2.2], [1.1, 2.2, 3.3]),
    ],
)
@pytest.mark.parametrize("null_policy_args", NULL_ARGS)
def test_factorize_nonnull(
    constructor_eager: ConstructorEager,
    *,
    values: list[Any],
    null_policy_args: dict[str, Any],
    expected_unique: int,
) -> None:
    if "polars" in str(constructor_eager) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    df_native = constructor_eager({"a": values})
    df = nw.from_native(df_native)
    encoded_result = df["a"].factorize(**null_policy_args)
    codes, uniqs = encoded_result

    reconstructed_values = {"a": [uniqs[i] if i >= 0 else None for i in codes]}
    assert_equal_data(df, reconstructed_values)
    assert uniqs.dtype == df["a"].dtype
    assert len(uniqs) == len(expected_unique)

    assert codes.dtype.is_integer()
    assert len(codes) == len(values)

    assert (codes >= 0).all()
    assert codes.name == "codes"
    assert uniqs.name == "uniques"

    assert_equal_series(
        encoded_result.uniques, [*encoded_result.mapping.keys()], name=uniqs.name
    )
    assert [*encoded_result.mapping.values()] == [*range(len(encoded_result.uniques))]


@pytest.mark.parametrize(
    ("values", "null_policy_args", "expected_unique"),
    [
        ([None], NULL_PRESERVE, []),
        ([None], NULL_ENCODE, [None]),
        ([None], NULL_SENTINEL, []),
        ([*"abcabc", None], NULL_PRESERVE, ["a", "b", "c"]),
        ([*"abcabc", None], NULL_ENCODE, ["a", "b", "c", None]),
        ([*"abcabc", None], NULL_SENTINEL, ["a", "b", "c"]),
        ([1, 2, 3, 2, None], NULL_PRESERVE, [1, 2, 3]),
        ([1, 2, 3, 2, None], NULL_ENCODE, [1, 2, 3, None]),
        ([1, 2, 3, 2, None], NULL_SENTINEL, [1, 2, 3]),
        ([1.1, 2.2, 3.3, 2.2, None], NULL_PRESERVE, [1.1, 2.2, 3.3]),
        ([1.1, 2.2, 3.3, 2.2, None], NULL_ENCODE, [1.1, 2.2, 3.3, None]),
        ([1.1, 2.2, 3.3, 2.2, None], NULL_SENTINEL, [1.1, 2.2, 3.3]),
        ([None, None], NULL_PRESERVE, []),
        ([None, None], NULL_ENCODE, [None]),
        ([None, None], NULL_SENTINEL, []),
        ([*"abcabc", None, None], NULL_PRESERVE, ["a", "b", "c"]),
        ([*"abcabc", None, None], NULL_ENCODE, ["a", "b", "c", None]),
        ([*"abcabc", None, None], NULL_SENTINEL, ["a", "b", "c"]),
        ([1, 2, 3, 2, None, None], NULL_PRESERVE, [1, 2, 3]),
        ([1, 2, 3, 2, None, None], NULL_ENCODE, [1, 2, 3, None]),
        ([1, 2, 3, 2, None, None], NULL_SENTINEL, [1, 2, 3]),
        ([1.1, 2.2, 3.3, 2.2, None, None], NULL_PRESERVE, [1.1, 2.2, 3.3]),
        ([1.1, 2.2, 3.3, 2.2, None, None], NULL_ENCODE, [1.1, 2.2, 3.3, None]),
        ([1.1, 2.2, 3.3, 2.2, None, None], NULL_SENTINEL, [1.1, 2.2, 3.3]),
    ],
)
def test_factorize_null(
    constructor_eager: ConstructorEager,
    *,
    values: list[Any],
    null_policy_args: dict[str, Any],
    expected_unique: int,
) -> None:
    if "polars" in str(constructor_eager) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    df_native = constructor_eager({"a": values})
    df = nw.from_native(df_native)
    encoded_result = df["a"].factorize(**null_policy_args)
    codes, uniqs = encoded_result

    reconstructed_values = {
        "a": [
            uniqs[i] if (not is_null and i >= 0) else None
            for i, is_null in zip(codes, codes.is_null(), strict=True)
        ]
    }
    assert_equal_data(df, reconstructed_values)
    assert uniqs.dtype == df["a"].dtype
    assert len(uniqs) == len(expected_unique)

    assert codes.dtype.is_integer()
    assert len(codes) == len(values)

    assert codes.name == "codes"
    assert uniqs.name == "uniques"

    assert_equal_series(
        encoded_result.uniques, [*encoded_result.mapping.keys()], name=uniqs.name
    )
    assert [*encoded_result.mapping.values()] == [*range(len(encoded_result.uniques))]


@pytest.mark.parametrize(
    ("values", "expected_uniqs", "expected_codes"),
    [
        ([], [], []),
        ([*"abcabc"], ["a", "b", "c"], [0, 1, 2, 0, 1, 2]),
        ([1, 2, 3, 2], [1, 2, 3], [0, 1, 2, 1]),
        ([1.1, 2.2, 3.3, 2.2], [1.1, 2.2, 3.3], [0, 1, 2, 1]),
    ],
)
@pytest.mark.parametrize("null_policy_args", NULL_ARGS)
def test_factorize_sort_nonnull(
    constructor_eager: ConstructorEager,
    *,
    values: list[Any],
    null_policy_args: dict[str, Any],
    expected_uniqs: list[Any],
    expected_codes: list[int],
) -> None:
    if "polars" in str(constructor_eager) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    df_native = constructor_eager({"a": values})
    df = nw.from_native(df_native)
    codes, uniqs = df["a"].factorize(sort=True, **null_policy_args)

    assert_equal_series(uniqs, expected_uniqs, name="uniques")
    assert_equal_series(codes, expected_codes, name="codes")


@pytest.mark.parametrize(
    ("values", "null_policy_args", "expected_uniqs", "expected_codes"),
    [
        ([None], NULL_PRESERVE, [], [None]),
        ([None], NULL_ENCODE, [None], [0]),
        ([None], NULL_SENTINEL, [], [NULL_SENTINEL["sentinel"]]),
        ([*"abcabc", None], NULL_PRESERVE, [*"abc"], [0, 1, 2, 0, 1, 2, None]),
        ([*"abcabc", None], NULL_ENCODE, [*"abc", None], [0, 1, 2, 0, 1, 2, 3]),
        (
            [*"abcabc", None],
            NULL_SENTINEL,
            [*"abc"],
            [0, 1, 2, 0, 1, 2, NULL_SENTINEL["sentinel"]],
        ),
        ([10, 11, 12, None], NULL_PRESERVE, [10, 11, 12], [0, 1, 2, None]),
        ([10, 11, 12, None], NULL_ENCODE, [10, 11, 12, None], [0, 1, 2, 3]),
        (
            [10, 11, 12, None],
            NULL_SENTINEL,
            [10, 11, 12],
            [0, 1, 2, NULL_SENTINEL["sentinel"]],
        ),
        ([None, None], NULL_PRESERVE, [], [None, None]),
        ([None, None], NULL_ENCODE, [None], [0, 0]),
        (
            [None, None],
            NULL_SENTINEL,
            [],
            [NULL_SENTINEL["sentinel"], NULL_SENTINEL["sentinel"]],
        ),
        (
            [*"abcabc", None, None],
            NULL_PRESERVE,
            [*"abc"],
            [0, 1, 2, 0, 1, 2, None, None],
        ),
        ([*"abcabc", None, None], NULL_ENCODE, [*"abc", None], [0, 1, 2, 0, 1, 2, 3, 3]),
        (
            [*"abcabc", None, None],
            NULL_SENTINEL,
            [*"abc"],
            [0, 1, 2, 0, 1, 2, NULL_SENTINEL["sentinel"], NULL_SENTINEL["sentinel"]],
        ),
        ([10, 11, 12, None, None], NULL_PRESERVE, [10, 11, 12], [0, 1, 2, None, None]),
        ([10, 11, 12, None, None], NULL_ENCODE, [10, 11, 12, None], [0, 1, 2, 3, 3]),
        (
            [10, 11, 12, None, None],
            NULL_SENTINEL,
            [10, 11, 12],
            [0, 1, 2, NULL_SENTINEL["sentinel"], NULL_SENTINEL["sentinel"]],
        ),
    ],
)
def test_factorize_sort_null(
    constructor_eager: ConstructorEager,
    *,
    values: list[Any],
    null_policy_args: dict[str, Any],
    expected_uniqs: list[Any],
    expected_codes: list[int],
) -> None:
    if "polars" in str(constructor_eager) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    df_native = constructor_eager({"a": values})
    df = nw.from_native(df_native)
    codes, uniqs = df["a"].factorize(sort=True, **null_policy_args)

    assert_equal_series(uniqs, expected_uniqs, name="uniques")
    assert_equal_series(codes, expected_codes, name="codes")


@pytest.mark.parametrize(
    ("values", "null_policy_args", "expected_unique", "expected_unique_pandas"),
    [
        (
            [1.1, 2.2, 1.1, float("nan")],
            NULL_PRESERVE,
            [1.1, 2.2, float("nan")],
            [1.1, 2.2],
        ),
        (
            [1.1, 2.2, 1.1, float("nan")],
            NULL_ENCODE,
            [1.1, 2.2, float("nan")],
            [1.1, 2.2, None],
        ),
        (
            [1.1, 2.2, 1.1, float("nan")],
            NULL_SENTINEL,
            [1.1, 2.2, float("nan")],
            [1.1, 2.2],
        ),
        (
            [1.1, 2.2, 1.1, float("nan"), None],
            NULL_PRESERVE,
            [1.1, 2.2, float("nan")],
            [1.1, 2.2],
        ),
        (
            [1.1, 2.2, 1.1, float("nan"), None],
            NULL_ENCODE,
            [1.1, 2.2, float("nan"), None],
            [1.1, 2.2, None],
        ),
        (
            [1.1, 2.2, 1.1, float("nan"), None],
            NULL_SENTINEL,
            [1.1, 2.2, float("nan")],
            [1.1, 2.2],
        ),
    ],
)
def test_factorize_nan_semantics(
    constructor_eager: ConstructorEager,
    *,
    values: list[float | None],
    null_policy_args: dict[str, Any],
    expected_unique: list[Any],
    expected_unique_pandas: list[Any],
) -> None:
    if "polars" in str(constructor_eager) and polars_lt_v1:
        pytest.skip(reason=pl_skip_reason)

    is_pandas_backend = any(x in str(constructor_eager) for x in ("pandas", "modin"))
    expected = expected_unique_pandas if is_pandas_backend else expected_unique

    df_native = constructor_eager({"a": values})
    df = nw.from_native(df_native)

    expected = df["a"].factorize(sort=True, **null_policy_args)
    assert_equal_series(expected.uniques, expected, name="uniques")
