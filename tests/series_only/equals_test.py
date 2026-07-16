from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager

TBD = None
_NAN_HANDLING = pytest.mark.xfail(
    strict=True, reason="NaN/null disambiguation not yet decided"
)
_CHECK_NAMES = pytest.mark.xfail(
    strict=True,
    reason=(
        "Bare ChunkedArray carries no name metadata, resulting in empty names - "
        "TBD whether check_names=True should raise for pyarrow in case of empty names"
    ),
)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        # --- Same dtype: int ---
        ([1, 2], [1, 2], True),
        # --- Same dtype: float ---
        ([1.1, 2.2], [1.1, 2.2], True),
        # --- Value differs ---
        ([1, 2], [1, 0], False),
        # --- Cross-supertype: float vs int ---
        ([1.0, 2.0], [1, 2], True),
        ([1.1, 2.2], [1, 2], False),
        # --- Cross-supertype: int vs string ---
        ([1, 2], ["1", "2"], False),
    ],
)
def test_series_equals(
    constructor_eager: ConstructorEager,
    left: list,
    right: list,
    expected: bool | None,  # noqa: FBT001
) -> None:
    left_native = constructor_eager({"left": left})["left"]
    right_native = constructor_eager({"right": right})["right"]
    left_nw = nw.from_native(left_native, series_only=True)
    right_nw = nw.from_native(right_native, series_only=True)

    result = left_nw.equals(right_nw)
    assert result == expected


@pytest.mark.parametrize(
    ("left", "right", "null_equal", "expected"),
    [
        # --- Null vs value ---
        ([1, None], [1, 2], True, False),
        ([1, None], [1, 2], False, False),
        # --- Null vs Null ---
        ([1, None], [1, None], True, True),
        ([1, None], [1, None], False, False),
        # --- NaN vs NaN ---
        # pandas: True, polars: True, pyarrow: False
        pytest.param(
            [1.0, float("nan")], [1.0, float("nan")], True, TBD, marks=_NAN_HANDLING
        ),
        # pandas: True, polars: True, pyarrow: False
        pytest.param(
            [1.0, float("nan")], [1.0, float("nan")], False, TBD, marks=_NAN_HANDLING
        ),
        # --- NaN vs value ---
        # pandas: False, polars: False, pyarrow: False
        ([1.0, float("nan")], [1.0, 2.0], True, False),
    ],
)
def test_series_equals_null_equal(
    constructor_eager: ConstructorEager,
    left: list,
    right: list,
    null_equal: bool,  # noqa: FBT001
    expected: bool | None,  # noqa: FBT001
) -> None:
    left_native = constructor_eager({"left": left})["left"]
    right_native = constructor_eager({"right": right})["right"]
    left_nw = nw.from_native(left_native, series_only=True)
    right_nw = nw.from_native(right_native, series_only=True)

    result = left_nw.equals(right_nw, null_equal=null_equal)
    assert result == expected


@pytest.mark.parametrize(
    ("left", "right", "check_dtypes", "expected"),
    [
        # --- Same values, different dtype ---
        ([1.1, 2.2], [1, 2], False, False),
        # --- Whole-number floats, check_dtypes=False ---
        ([1.0, 2.0], [1, 2], False, True),
        # --- Whole-number floats, check_dtypes=True ---
        ([1.0, 2.0], [1, 2], True, False),
    ],
)
def test_series_equals_check_dtypes(
    constructor_eager: ConstructorEager,
    left: list,
    right: list,
    check_dtypes: bool,  # noqa: FBT001
    expected: bool | None,  # noqa: FBT001
) -> None:
    left_native = constructor_eager({"a": left})["a"]
    right_native = constructor_eager({"a": right})["a"]
    left_nw = nw.from_native(left_native, series_only=True)
    right_nw = nw.from_native(right_native, series_only=True)

    result = left_nw.equals(right_nw, check_dtypes=check_dtypes)
    assert result == expected


@pytest.mark.parametrize(
    ("left_name", "right_name", "check_names", "expected"),
    [
        # --- Different names ---
        ("left", "right", False, True),
        # pandas: False, polars: False, pyarrow: True (ChunkedArray has no name metadata)
        pytest.param("left", "right", True, TBD, marks=_CHECK_NAMES),
    ],
)
def test_series_equals_check_names(
    constructor_eager: ConstructorEager,
    left_name: str,
    right_name: str,
    check_names: bool,  # noqa: FBT001
    expected: bool | None,  # noqa: FBT001
) -> None:
    data = [1, 2]
    left_native = constructor_eager({left_name: data})[left_name]
    right_native = constructor_eager({right_name: data})[right_name]
    left_nw = nw.from_native(left_native, series_only=True)
    right_nw = nw.from_native(right_native, series_only=True)

    result = left_nw.equals(right_nw, check_names=check_names)
    assert result == expected
