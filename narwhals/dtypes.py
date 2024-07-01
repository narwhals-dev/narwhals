from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals.dependencies import get_polars
from narwhals.utils import isinstance_or_issubclass

if TYPE_CHECKING:
    from typing_extensions import Self


class DType:
    def __repr__(self) -> str:  # pragma: no cover
        return self.__class__.__qualname__

    @classmethod
    def is_numeric(cls: type[Self]) -> bool:
        return issubclass(cls, NumericType)

    def __eq__(self, other: DType | type[DType]) -> bool:  # type: ignore[override]
        return isinstance_or_issubclass(other, type(self))


class NumericType(DType): ...


class TemporalType(DType): ...


class Int64(NumericType): ...


class Int32(NumericType): ...


class Int16(NumericType): ...


class Int8(NumericType): ...


class UInt64(NumericType): ...


class UInt32(NumericType): ...


class UInt16(NumericType): ...


class UInt8(NumericType): ...


class Float64(NumericType): ...


class Float32(NumericType): ...


class String(DType): ...


class Boolean(DType): ...


class Object(DType): ...


class Datetime(TemporalType): ...


class Duration(TemporalType): ...


class Categorical(DType): ...


class Date(TemporalType): ...


def translate_dtype(plx: Any, dtype: type[DType]) -> Any:
    if "polars" in str(type(dtype)):
        msg = (
            f"Expected Narwhals object, got: {type(dtype)}.\n\n"
            "Perhaps you:\n"
            "- Forgot a `nw.from_native` somewhere?\n"
            "- Used `pl.Int64` instead of `nw.Int64`?"
        )
        raise TypeError(msg)

    dtype_mapping = {
        Float64: plx.Float64,
        Float32: plx.Float32,
        Int64: plx.Int64,
        Int32: plx.Int32,
        Int16: plx.Int16,
        Int8: plx.Int8,
        UInt64: plx.UInt64,
        UInt32: plx.UInt32,
        UInt16: plx.UInt16,
        UInt8: plx.UInt8,
        String: plx.String,
        Boolean: plx.Boolean,
        Categorical: plx.Categorical,
        Datetime: plx.Datetime,
        Duration: plx.Duration,
        Date: plx.Date,
    }

    if dtype in dtype_mapping:
        return dtype_mapping[dtype]

    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def to_narwhals_dtype(dtype: Any, *, is_polars: bool) -> DType:
    if not is_polars:
        return dtype  # type: ignore[no-any-return]
    pl = get_polars()

    dtype_mapping = {
        pl.Float64: Float64,
        pl.Float32: Float32,
        pl.Int64: Int64,
        pl.Int32: Int32,
        pl.Int16: Int16,
        pl.Int8: Int8,
        pl.UInt64: UInt64,
        pl.UInt32: UInt32,
        pl.UInt16: UInt16,
        pl.UInt8: UInt8,
        pl.String: String,
        pl.Boolean: Boolean,
        pl.Object: Object,
        pl.Categorical: Categorical,
        pl.Datetime: Datetime,
        pl.Duration: Duration,
        pl.Date: Date,
    }

    dtype_type = type(dtype)
    if dtype_type in dtype_mapping:
        return dtype_mapping[dtype_type]()

    msg = f"Unexpected dtype, got: {dtype_type}"  # pragma: no cover
    raise AssertionError(msg)
