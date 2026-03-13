from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._utils import inherit_doc
from narwhals.dtypes._classes import (
    Datetime as NwDatetime,
    DType,
    DTypeClass,
    Duration as NwDuration,
    Enum as NwEnum,
)

if TYPE_CHECKING:
    from datetime import timezone

    from narwhals.typing import TimeUnit

__all__ = ["Datetime", "Duration", "Enum"]


class Datetime(NwDatetime):
    @inherit_doc(NwDatetime)
    def __init__(
        self, time_unit: TimeUnit = "us", time_zone: str | timezone | None = None
    ) -> None:
        super().__init__(time_unit, time_zone)

    def __hash__(self) -> int:
        return hash(self.__class__)


class Duration(NwDuration):
    @inherit_doc(NwDuration)
    def __init__(self, time_unit: TimeUnit = "us") -> None:
        super().__init__(time_unit)

    def __hash__(self) -> int:
        return hash(self.__class__)


class Enum(NwEnum):
    """A fixed categorical encoding of a unique set of strings.

    Polars has an Enum data type, while pandas and PyArrow do not.

    Examples:
       >>> import polars as pl
       >>> import narwhals.stable.v1 as nw
       >>> data = ["beluga", "narwhal", "orca"]
       >>> s_native = pl.Series(data, dtype=pl.Enum(data))
       >>> nw.from_native(s_native, series_only=True).dtype
       Enum
    """

    def __init__(self) -> None:
        super(NwEnum, self).__init__()

    def __eq__(self, other: DType | type[DType]) -> bool:  # type: ignore[override]
        if type(other) is DTypeClass:
            return other in {type(self), NwEnum}
        return isinstance(other, type(self))

    def __hash__(self) -> int:  # pragma: no cover
        return super(NwEnum, self).__hash__()

    def __repr__(self) -> str:  # pragma: no cover
        return super(NwEnum, self).__repr__()
