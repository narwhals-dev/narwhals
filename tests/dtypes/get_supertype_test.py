from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from narwhals.dtypes._supertyping import get_supertype
from tests.utils import dtype_ids

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import TypeAlias

    from narwhals.dtypes import DType, NumericType, TemporalType
    from narwhals.typing import IntoDType

    IntoStruct: TypeAlias = Mapping[str, IntoDType]


def _check_supertype(left: DType, right: DType, expected: DType | None) -> None:
    result = get_supertype(left, right)
    if expected is None:
        assert result is None
    else:
        assert result is not None
        assert result == expected


@pytest.mark.parametrize(
    "dtype",
    [
        nw.Array(nw.Binary(), shape=(2,)),
        nw.Array(nw.Boolean, shape=(2, 3)),
        nw.Binary(),
        nw.Boolean(),
        nw.Categorical(),
        nw.Date(),
        nw.Datetime(),
        nw.Datetime(time_unit="ns", time_zone="Europe/Berlin"),
        nw.Decimal(),
        nw.Duration(),
        nw.Enum(["orca", "narwhal"]),
        nw.Enum([]),
        nw.Float32(),
        nw.Float64(),
        nw.Int8(),
        nw.Int16(),
        nw.Int32(),
        nw.Int64(),
        nw.Int128(),
        nw.List(nw.String),
        nw.List(nw.Array(nw.Int32, shape=(5, 3))),
        nw.Object(),
        nw.String(),
        nw.Struct({"r2": nw.Float64(), "mse": nw.Float32()}),
        nw.Struct({"a": nw.String, "b": nw.List(nw.Int32)}),
        nw.Time(),
        nw.UInt8(),
        nw.UInt16(),
        nw.UInt32(),
        nw.UInt64(),
        nw.UInt128(),
        nw.Unknown(),
    ],
    ids=dtype_ids,
)
def test_identical_dtype(dtype: DType) -> None:
    _check_supertype(dtype, dtype, dtype)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (nw.Datetime("ns"), nw.Datetime("us"), nw.Datetime("us")),
        (nw.Datetime("s"), nw.Datetime("us"), nw.Datetime("s")),
        (nw.Datetime("s"), nw.Datetime("s", "Africa/Accra"), None),
        (nw.Datetime(time_zone="Asia/Kathmandu"), nw.Datetime(), None),
        (
            nw.Enum(["beluga", "narwhal", "orca"]),
            nw.Enum(["dog", "cat", "fish with legs"]),
            None,
        ),
        (nw.Enum([]), nw.Enum(["fruit", "other food"]), None),
        (nw.List(nw.Int64), nw.List(nw.Int64()), nw.List(nw.Int64())),
        (nw.List(nw.UInt16()), nw.List(nw.Int32), nw.List(nw.Int32())),
        (nw.List(nw.Date), nw.List(nw.Binary), None),
        (nw.List(nw.Unknown), nw.List(nw.Float64), nw.List(nw.Unknown())),
        (
            nw.Array(nw.Float32, shape=2),
            nw.Array(nw.Float64, shape=2),
            nw.Array(nw.Float64, shape=2),
        ),
        (nw.Array(nw.Int64, shape=1), nw.Array(nw.Int64, shape=4), None),
        (
            nw.Array(nw.Decimal, shape=1),
            nw.Array(nw.Unknown, shape=1),
            nw.Array(nw.Unknown, shape=1),
        ),
        (
            nw.Array(nw.UInt128, shape=3),
            nw.Array(nw.Decimal, shape=3),
            nw.Array(nw.Decimal, shape=3),
        ),
        (nw.Array(nw.String, shape=1), nw.Array(nw.Int64, shape=1), None),
    ],
    ids=dtype_ids,
)
def test_same_class(left: DType, right: DType, expected: DType | None) -> None:
    _check_supertype(left, right, expected)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (
            {"f0": nw.Duration("ms"), "f1": nw.Int64, "f2": nw.Int64},
            {"f0": nw.Duration("us"), "f1": nw.Int64()},
            {"f0": nw.Duration("ms"), "f1": nw.Int64(), "f2": nw.Int64()},
        ),
        (
            {"f0": nw.Float64, "f1": nw.Date, "f2": nw.Int32},
            {"f0": nw.Float32, "f1": nw.Datetime, "f3": nw.UInt8},
            {"f0": nw.Float64, "f1": nw.Datetime(), "f2": nw.Int32, "f3": nw.UInt8},
        ),
        (
            {"f0": nw.Int32, "f1": nw.Boolean, "f2": nw.String},
            {"f0": nw.Unknown},
            {"f0": nw.Unknown, "f1": nw.Boolean, "f2": nw.String},
        ),
        (
            {"f0": nw.Object, "f1": nw.List(nw.Boolean)},
            {"f0": nw.List(nw.Boolean), "f1": nw.List(nw.Boolean)},
            None,
        ),
        ({"f0": nw.Binary()}, {"f0": nw.Datetime("s"), "f1": nw.Date}, None),
        (
            {"f0": nw.Int64, "f1": nw.Struct({"f1": nw.Float32, "f0": nw.String})},
            {
                "f0": nw.UInt8,
                "f1": nw.Struct(
                    {"f0": nw.Categorical, "f1": nw.Float64(), "f2": nw.Time}
                ),
            },
            {
                "f0": nw.Int64,
                "f1": nw.Struct({"f0": nw.String, "f1": nw.Float64, "f2": nw.Time}),
            },
        ),
        (
            {"F0": nw.UInt8, "f0": nw.Int16},
            {"f0": nw.Int128, "f1": nw.UInt16, " f0": nw.Int8},
            {"f0": nw.Int128, "f1": nw.UInt16, " f0": nw.Int8, "F0": nw.UInt8},
        ),
    ],
    ids=dtype_ids,
)
def test_struct(left: IntoStruct, right: IntoStruct, expected: IntoStruct | None) -> None:
    expected_ = None if expected is None else nw.Struct(expected)
    _check_supertype(nw.Struct(left), nw.Struct(right), expected_)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (nw.Datetime("ns"), nw.Date(), nw.Datetime("ns")),
        (nw.Date(), nw.Datetime(), nw.Datetime()),
        (nw.Datetime(), nw.Int8(), None),
        (nw.String(), nw.Categorical(), nw.String()),
        (nw.Enum(["hello"]), nw.Categorical(), None),
        (nw.Enum(["hello"]), nw.String(), nw.String()),
        (nw.Binary(), nw.String(), nw.Binary()),
    ],
    ids=dtype_ids,
)
def test_mixed_dtype(left: DType, right: DType, expected: DType | None) -> None:
    _check_supertype(left, right, expected)


def test_mixed_integer_temporal(
    naive_temporal_dtype: TemporalType, numeric_dtype: NumericType
) -> None:
    _check_supertype(naive_temporal_dtype, numeric_dtype, None)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        # NOTE: The order of the case *should not* matter (some are flipped for coverage)
        # signed + signed
        (nw.Int8(), nw.Int16(), nw.Int16()),
        (nw.Int8(), nw.Int32(), nw.Int32()),
        (nw.Int8(), nw.Int64(), nw.Int64()),
        (nw.Int8(), nw.Int128(), nw.Int128()),
        (nw.Int16(), nw.Int32(), nw.Int32()),
        (nw.Int64(), nw.Int16(), nw.Int64()),
        (nw.Int16(), nw.Int128(), nw.Int128()),
        (nw.Int64(), nw.Int32(), nw.Int64()),
        (nw.Int32(), nw.Int128(), nw.Int128()),
        (nw.Int64(), nw.Int128(), nw.Int128()),
        # unsigned + unsigned
        (nw.UInt8(), nw.UInt16(), nw.UInt16()),
        (nw.UInt32(), nw.UInt8(), nw.UInt32()),
        (nw.UInt8(), nw.UInt64(), nw.UInt64()),
        (nw.UInt8(), nw.UInt128(), nw.UInt128()),
        (nw.UInt16(), nw.UInt32(), nw.UInt32()),
        (nw.UInt16(), nw.UInt64(), nw.UInt64()),
        (nw.UInt128(), nw.UInt16(), nw.UInt128()),
        (nw.UInt32(), nw.UInt64(), nw.UInt64()),
        (nw.UInt32(), nw.UInt128(), nw.UInt128()),
        (nw.UInt64(), nw.UInt128(), nw.UInt128()),
        # signed + unsigned
        (nw.Int8(), nw.UInt8(), nw.Int16()),
        (nw.Int8(), nw.UInt16(), nw.Int32()),
        (nw.UInt32(), nw.Int8(), nw.Int64()),
        (nw.Int8(), nw.UInt64(), nw.Float64()),
        (nw.Int16(), nw.UInt8(), nw.Int16()),
        (nw.Int16(), nw.UInt16(), nw.Int32()),
        (nw.UInt32(), nw.Int16(), nw.Int64()),
        (nw.Int16(), nw.UInt64(), nw.Float64()),
        (nw.Int32(), nw.UInt8(), nw.Int32()),
        (nw.UInt16(), nw.Int32(), nw.Int32()),
        (nw.Int32(), nw.UInt32(), nw.Int64()),
        (nw.Int32(), nw.UInt64(), nw.Float64()),
        (nw.Int64(), nw.UInt8(), nw.Int64()),
        (nw.UInt16(), nw.Int64(), nw.Int64()),
        (nw.Int64(), nw.UInt32(), nw.Int64()),
        (nw.Int64(), nw.UInt64(), nw.Float64()),
        # float + float
        (nw.Float32(), nw.Float64(), nw.Float64()),
        (nw.Float64(), nw.Float32(), nw.Float64()),
        # float + integer
        (nw.Int8(), nw.Float32(), nw.Float32()),
        (nw.Int16(), nw.Float32(), nw.Float32()),
        (nw.Float32(), nw.UInt8(), nw.Float32()),
        (nw.Float32(), nw.UInt16(), nw.Float32()),
        (nw.Int32(), nw.Float32(), nw.Float64()),
        (nw.Int64(), nw.Float32(), nw.Float64()),
        (nw.UInt32(), nw.Float32(), nw.Float64()),
        (nw.Float32(), nw.UInt64(), nw.Float64()),
        (nw.Int8(), nw.Float64(), nw.Float64()),
        (nw.Float64(), nw.Int64(), nw.Float64()),
        # float + decimal
        (nw.Decimal(), nw.Float32(), nw.Float64()),
        (nw.Decimal(), nw.Float64(), nw.Float64()),
        # decimal + decimal
        (nw.Decimal(5, 2), nw.Decimal(4, 3), nw.Decimal(5, 3)),
        (nw.Decimal(scale=12), nw.Decimal(18, scale=9), nw.Decimal(38, 12)),
        # decimal + integer
        (nw.Decimal(4, 1), nw.UInt8(), nw.Decimal(4, 1)),
        (nw.Decimal(5, 2), nw.Int8(), nw.Decimal(5, 2)),
        (nw.Decimal(10, 0), nw.Int32(), nw.Decimal(10, 0)),
        (nw.Decimal(15, 2), nw.UInt32(), nw.Decimal(15, 2)),
        (nw.Decimal(2, 1), nw.UInt8, nw.Decimal(38, 1)),
        (nw.Decimal(10, 5), nw.Int64, nw.Decimal(38, 5)),
        (nw.Decimal(38, 0), nw.Int128, nw.Decimal(38, 0)),
        (nw.Decimal(1, 0), nw.UInt8(), nw.Decimal(38, 0)),
        (nw.Decimal(38, 38), nw.Int8(), nw.Decimal(38, 38)),
        (nw.Decimal(10, 1), nw.UInt32(), nw.Decimal(38, 1)),
    ],
    ids=dtype_ids,
)
def test_numeric_promotion(left: DType, right: DType, expected: DType) -> None:
    _check_supertype(left, right, expected)
    _check_supertype(right, left, expected)


def test_numeric_and_bool_promotion(numeric_dtype: NumericType) -> None:
    _check_supertype(numeric_dtype, nw.Boolean(), numeric_dtype)
    _check_supertype(nw.Boolean(), numeric_dtype, numeric_dtype)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (nw_v1.Datetime(), nw_v1.Datetime(), nw_v1.Datetime()),
        (nw_v1.Datetime("ns"), nw_v1.Datetime("s"), nw_v1.Datetime("s")),
        (
            nw_v1.Datetime(time_zone="Europe/Berlin"),
            nw_v1.Datetime(time_zone="Europe/Berlin"),
            nw_v1.Datetime(time_zone="Europe/Berlin"),
        ),
        (
            nw_v1.Datetime(time_zone="Europe/Berlin"),
            nw_v1.Datetime("ms", "Europe/Berlin"),
            nw_v1.Datetime("ms", "Europe/Berlin"),
        ),
        (nw_v1.Datetime(time_zone="Europe/Berlin"), nw_v1.Datetime(), None),
        (nw_v1.Datetime("s"), nw_v1.Datetime("s", "Africa/Accra"), None),
        (nw_v1.Duration("ns"), nw_v1.Duration("ms"), nw_v1.Duration("ms")),
        (nw_v1.Duration(), nw_v1.Duration(), nw_v1.Duration()),
        (nw_v1.Duration("s"), nw_v1.Duration(), nw_v1.Duration("s")),
        (nw_v1.Duration(), nw_v1.Datetime(), None),
        (nw_v1.Enum(), nw_v1.Enum(), nw_v1.Enum()),
        (nw_v1.Enum(), nw_v1.String(), nw_v1.String()),
        (
            nw.Date(),
            nw_v1.Datetime(time_zone="Europe/Berlin"),
            nw_v1.Datetime(time_zone="Europe/Berlin"),
        ),
        (
            nw.Struct({"f0": nw_v1.Duration("ms"), "f1": nw.Int64, "f2": nw.Int64}),
            nw.Struct({"f0": nw_v1.Duration("us"), "f1": nw.Int64()}),
            nw.Struct({"f0": nw_v1.Duration("ms"), "f1": nw.Int64(), "f2": nw.Int64()}),
        ),
        (
            nw.Struct({"f0": nw.Float64, "f1": nw.Date, "f2": nw.Int32}),
            nw.Struct({"f0": nw.Float32, "f1": nw_v1.Datetime, "f3": nw.UInt8}),
            nw.Struct(
                {"f0": nw.Float64, "f1": nw_v1.Datetime(), "f2": nw.Int32, "f3": nw.UInt8}
            ),
        ),
        (
            nw.Struct({"f0": nw.Binary()}),
            nw.Struct({"f0": nw_v1.Datetime("s"), "f1": nw.Date}),
            None,
        ),
        (
            nw.Array(nw.Date, shape=3),
            nw.Array(nw_v1.Datetime, shape=3),
            nw.Array(nw_v1.Datetime, shape=3),
        ),
        (nw.Array(nw.Date, shape=1), nw.Array(nw_v1.Datetime, shape=3), None),
        (nw.Array(nw.Date, shape=2), nw.Array(nw_v1.Duration, shape=2), None),
        (
            nw.Array(nw_v1.Enum(), shape=4),
            nw.Array(nw_v1.Enum(), shape=4),
            nw.Array(nw_v1.Enum(), shape=4),
        ),
    ],
    ids=dtype_ids,
)
def test_v1_dtypes(left: DType, right: DType, expected: DType | None) -> None:
    result = get_supertype(left, right)
    if expected is None:
        assert result is None
    else:
        assert result is not None
        assert result == expected
        # Must also preserve v1-ness
        assert type(result) is type(expected)


@pytest.mark.parametrize(
    ("dtype_v1", "dtype_main"),
    [
        (nw_v1.Duration("ms"), nw.Duration("ms")),
        (nw_v1.Duration("ns"), nw.Duration("ns")),
        (nw_v1.Datetime(time_unit="ms"), nw.Datetime(time_unit="ms")),
        (nw_v1.Datetime(time_zone="Europe/Rome"), nw.Datetime(time_zone="Europe/Rome")),
        (nw_v1.Enum(), nw.Enum([])),
    ],
)
def test_mixed_versions_return_none(dtype_v1: DType, dtype_main: DType) -> None:
    _check_supertype(dtype_v1, dtype_main, None)
    _check_supertype(dtype_main, dtype_v1, None)
