from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals.spec import DType as DTypeProtocol

if TYPE_CHECKING:
    from typing_extensions import Self


class DType(DTypeProtocol):
    def __repr__(self) -> str:
        return self.__class__.__qualname__

    @classmethod
    def is_numeric(cls: Self) -> bool:
        return issubclass(cls, NumericType)


class NumericType(DType):
    ...


class Int64(NumericType):
    ...


class Int32(NumericType):
    ...


class Int16(NumericType):
    ...


class Int8(NumericType):
    ...


class UInt64(NumericType):
    ...


class UInt32(NumericType):
    ...


class UInt16(NumericType):
    ...


class UInt8(NumericType):
    ...


class Float64(NumericType):
    ...


class Float32(NumericType):
    ...


class String(DType):
    ...


class Bool(DType):
    ...
