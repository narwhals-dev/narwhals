from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

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


class Datetime(TemporalType): ...


class Date(TemporalType): ...


def translate_dtype(plx: Any, dtype: DType) -> Any:
    if "polars" in str(type(dtype)):
        msg = (
            f"Expected Narwhals object, got: {type(dtype)}.\n\n"
            "Perhaps you:\n"
            "- Forgot a `nw.from_native` somewhere?\n"
            "- Used `pl.Int64` instead of `nw.Int64`?"
        )
        raise TypeError(msg)
    if dtype == Float64:
        return plx.Float64
    if dtype == Float32:
        return plx.Float32
    if dtype == Int64:
        return plx.Int64
    if dtype == Int32:
        return plx.Int32
    if dtype == Int16:
        return plx.Int16
    if dtype == Int8:
        return plx.Int8
    if dtype == UInt64:
        return plx.UInt64
    if dtype == UInt32:
        return plx.UInt32
    if dtype == UInt16:
        return plx.UInt16
    if dtype == UInt8:
        return plx.UInt8
    if dtype == String:
        return plx.String
    if dtype == Boolean:
        return plx.Boolean
    if dtype == Datetime:
        return plx.Datetime
    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def to_narwhals_dtype(dtype: Any, *, is_polars: bool) -> DType:
    if not is_polars:
        return dtype  # type: ignore[no-any-return]
    import polars as pl

    if dtype == pl.Float64:
        return Float64()
    if dtype == pl.Float32:
        return Float32()
    if dtype == pl.Int64:
        return Int64()
    if dtype == pl.Int32:
        return Int32()
    if dtype == pl.Int16:
        return Int16()
    if dtype == pl.Int8:
        return Int8()
    if dtype == pl.UInt64:
        return UInt64()
    if dtype == pl.UInt32:
        return UInt32()
    if dtype == pl.UInt16:
        return UInt16()
    if dtype == pl.UInt8:
        return UInt8()
    if dtype == pl.String:
        return String()
    if dtype == pl.Boolean:
        return Boolean()
    if dtype == pl.Datetime:
        return Datetime()
    msg = f"Unexpected dtype, got: {type(dtype)}"  # pragma: no cover
    raise AssertionError(msg)
