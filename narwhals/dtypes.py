from __future__ import annotations

from datetime import timezone
from typing import TYPE_CHECKING
from typing import Literal

if TYPE_CHECKING:
    from typing_extensions import Self


class DType:
    def __repr__(self) -> str:  # pragma: no cover
        return self.__class__.__qualname__

    @classmethod
    def is_numeric(cls: type[Self]) -> bool:
        return issubclass(cls, NumericType)

    def __eq__(self, other: DType | type[DType]) -> bool:  # type: ignore[override]
        from narwhals.utils import isinstance_or_issubclass

        return isinstance_or_issubclass(other, type(self))

    def __hash__(self) -> int:
        return hash(self.__class__)


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


class Unknown(DType): ...


class Datetime(TemporalType):
    """
    Data type representing a calendar date and time of day.

    Arguments:
        time_unit: Unit of time. Defaults to `'us'` (microseconds).
        time_zone: Time zone string, as defined in zoneinfo (to see valid strings run
            `import zoneinfo; zoneinfo.available_timezones()` for a full list).
            When used to match dtypes, can set this to "*" to check for Datetime
            columns that have any (non-null) timezone.

    Notes:
        Adapted from Polars implementation at:
        https://github.com/pola-rs/polars/blob/py-1.7.1/py-polars/polars/datatypes/classes.py#L398-L457
    """

    def __init__(
        self: Self,
        time_unit: Literal["us", "ns", "ms"] = "us",
        time_zone: str | timezone | None = None,
    ) -> None:
        if time_unit not in {"ms", "us", "ns"}:
            msg = (
                "invalid `time_unit`"
                f"\n\nExpected one of {{'ns','us','ms'}}, got {time_unit!r}."
            )
            raise ValueError(msg)

        if isinstance(time_zone, timezone):
            time_zone = str(time_zone)

        self.time_unit = time_unit
        self.time_zone = time_zone

    def __eq__(self: Self, other: object) -> bool:
        # allow comparing object instances to class
        if type(other) is type and issubclass(other, Datetime):
            return True
        elif isinstance(other, Datetime):
            return self.time_unit == other.time_unit and self.time_zone == other.time_zone
        else:
            return False

    def __hash__(self: Self) -> int:  # pragma: no cover
        return hash((self.__class__, self.time_unit, self.time_zone))

    def __repr__(self: Self) -> str:  # pragma: no cover
        class_name = self.__class__.__name__
        return f"{class_name}(time_unit={self.time_unit!r}, time_zone={self.time_zone!r})"


class Duration(TemporalType):
    """
    Data type representing a time duration.

    Arguments:
        time_unit: Unit of time. Defaults to `'us'` (microseconds).

    Notes:
        Adapted from Polars implementation at:
        https://github.com/pola-rs/polars/blob/py-1.7.1/py-polars/polars/datatypes/classes.py#L460-L502
    """

    def __init__(
        self: Self,
        time_unit: Literal["us", "ns", "ms"] = "us",
    ) -> None:
        if time_unit not in ("ms", "us", "ns"):
            msg = (
                "invalid `time_unit`"
                f"\n\nExpected one of {{'ns','us','ms'}}, got {time_unit!r}."
            )
            raise ValueError(msg)

        self.time_unit = time_unit

    def __eq__(self: Self, other: object) -> bool:
        # allow comparing object instances to class
        if type(other) is type and issubclass(other, Duration):
            return True
        elif isinstance(other, Duration):
            return self.time_unit == other.time_unit
        else:
            return False

    def __hash__(self: Self) -> int:  # pragma: no cover
        return hash((self.__class__, self.time_unit))

    def __repr__(self: Self) -> str:  # pragma: no cover
        class_name = self.__class__.__name__
        return f"{class_name}(time_unit={self.time_unit!r})"


class Categorical(DType): ...


class Enum(DType): ...


class Date(TemporalType): ...
