from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals._utils import (
    Version,
    check_comparable_dtype,
    check_comparable_values,
    is_comparable_dtype,
    python_literal_dtype,
)
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from narwhals.dtypes import DType
    from narwhals.typing import PythonLiteral

DTYPES = Version.MAIN.dtypes


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (True, nw.Boolean()),
        (1, nw.Int64()),
        (1.5, nw.Float64()),
        ("a", nw.String()),
        (Decimal("1.5"), nw.Decimal()),
        (b"a", nw.Binary()),
        (bytearray(b"a"), nw.Binary()),
        (dt.datetime(2020, 1, 1), nw.Datetime()),
        (dt.date(2020, 1, 1), nw.Date()),
        (dt.time(1, 2), nw.Time()),
        (dt.timedelta(days=1), nw.Duration()),
        ([1, 2], nw.List(nw.Unknown())),
        ({1, 2}, nw.List(nw.Unknown())),
        ({"a": 1}, nw.List(nw.Unknown())),
    ],
)
def test_python_literal_dtype(value: PythonLiteral, expected: DType | None) -> None:
    assert python_literal_dtype(value, DTYPES) == expected


def test_python_literal_dtype_unknown() -> None:
    assert python_literal_dtype(object(), DTYPES) is None  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("dtype", "other"),
    [
        (nw.Int64(), nw.Int64()),
        (nw.Int64(), nw.UInt8()),
        (nw.Float32(), nw.Float64()),
        # `Decimal` is exact, so it is comparable against any other numeric dtype.
        (nw.Decimal(), nw.Int64()),
        (nw.Int64(), nw.Decimal()),
        # `String`, `Categorical` and `Enum` are interchangeable.
        (nw.String(), nw.Categorical()),
        (nw.Enum(["a", "b"]), nw.String()),
        # Time units and inner dtypes are not part of the comparison.
        (nw.Datetime("ms"), nw.Datetime("us")),
        (nw.Duration("ns"), nw.Duration("us")),
        (nw.List(nw.Int64()), nw.List(nw.Unknown())),
        (nw.Date(), nw.Date()),
        (nw.Time(), nw.Time()),
        (nw.Binary(), nw.Binary()),
        (nw.Boolean(), nw.Boolean()),
        # Nothing is known about `Object`/`Unknown`, so no validation is performed.
        (nw.Object(), nw.Int64()),
        (nw.Int64(), nw.Unknown()),
    ],
)
def test_is_comparable_dtype(dtype: DType, other: DType) -> None:
    assert is_comparable_dtype(dtype, other, DTYPES)
    check_comparable_dtype("is_in", dtype, other, DTYPES)


@pytest.mark.parametrize(
    ("dtype", "other"),
    [
        (nw.Int64(), nw.Float64()),
        (nw.Float64(), nw.Int64()),
        (nw.Int64(), nw.Boolean()),
        (nw.Boolean(), nw.Int64()),
        (nw.Int64(), nw.String()),
        (nw.String(), nw.Int64()),
        (nw.String(), nw.Binary()),
        (nw.Date(), nw.Datetime()),
        (nw.Datetime(), nw.Duration()),
        (nw.Int64(), nw.List(nw.Unknown())),
        (nw.List(nw.Int64()), nw.Struct({"a": nw.Int64()})),
    ],
)
def test_is_comparable_dtype_false(dtype: DType, other: DType) -> None:
    assert not is_comparable_dtype(dtype, other, DTYPES)


def test_check_comparable_dtype_raises() -> None:
    with pytest.raises(
        InvalidOperationError, match="cannot check for Float64 values in Int64 data"
    ):
        check_comparable_dtype("is_in", nw.Int64(), nw.Float64(), DTYPES)


@pytest.mark.parametrize(
    "values",
    [
        [],
        [None],
        [None, 1, 2],
        # Values Narwhals can't classify are left to the backend to deal with.
        [object()],
    ],
)
def test_check_comparable_values(values: list[PythonLiteral]) -> None:
    check_comparable_values("is_in", nw.Int64(), values, DTYPES)


def test_check_comparable_values_raises() -> None:
    # A collection's dtype is inferred from its first non-null element.
    with pytest.raises(
        InvalidOperationError, match="cannot check for String values in Int64 data"
    ):
        check_comparable_values("is_in", nw.Int64(), [None, "a", 1], DTYPES)
