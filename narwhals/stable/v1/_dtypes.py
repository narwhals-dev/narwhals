from __future__ import annotations

from narwhals.dtypes import Array
from narwhals.dtypes import Binary
from narwhals.dtypes import Boolean
from narwhals.dtypes import Categorical
from narwhals.dtypes import Date
from narwhals.dtypes import Datetime as NwDatetime
from narwhals.dtypes import Decimal
from narwhals.dtypes import DType
from narwhals.dtypes import Duration as NwDuration
from narwhals.dtypes import Enum as NwEnum
from narwhals.dtypes import Field
from narwhals.dtypes import Float32
from narwhals.dtypes import Float64
from narwhals.dtypes import FloatType
from narwhals.dtypes import Int8
from narwhals.dtypes import Int16
from narwhals.dtypes import Int32
from narwhals.dtypes import Int64
from narwhals.dtypes import Int128
from narwhals.dtypes import IntegerType
from narwhals.dtypes import List
from narwhals.dtypes import NestedType
from narwhals.dtypes import NumericType
from narwhals.dtypes import Object
from narwhals.dtypes import SignedIntegerType
from narwhals.dtypes import String
from narwhals.dtypes import Struct
from narwhals.dtypes import Time
from narwhals.dtypes import UInt8
from narwhals.dtypes import UInt16
from narwhals.dtypes import UInt32
from narwhals.dtypes import UInt64
from narwhals.dtypes import UInt128
from narwhals.dtypes import Unknown
from narwhals.dtypes import UnsignedIntegerType


class Datetime(NwDatetime):
    """Data type representing a calendar date and time of day.

    Arguments:
        time_unit: Unit of time. Defaults to `'us'` (microseconds).
        time_zone: Time zone string, as defined in zoneinfo (to see valid strings run
            `import zoneinfo; zoneinfo.available_timezones()` for a full list).

    Notes:
        Adapted from [Polars implementation](https://github.com/pola-rs/polars/blob/py-1.7.1/py-polars/polars/datatypes/classes.py#L398-L457)
    """

    def __hash__(self) -> int:
        return hash(self.__class__)


class Duration(NwDuration):
    """Data type representing a time duration.

    Arguments:
        time_unit: Unit of time. Defaults to `'us'` (microseconds).

    Notes:
        Adapted from [Polars implementation](https://github.com/pola-rs/polars/blob/py-1.7.1/py-polars/polars/datatypes/classes.py#L460-L502)
    """

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
        if type(other) is type:
            return other in {type(self), NwEnum}
        return isinstance(other, type(self))

    def __hash__(self) -> int:  # pragma: no cover
        return super(NwEnum, self).__hash__()

    def __repr__(self) -> str:  # pragma: no cover
        return super(NwEnum, self).__repr__()


__all__ = [
    "Array",
    "Binary",
    "Boolean",
    "Categorical",
    "DType",
    "Date",
    "Datetime",
    "Decimal",
    "Duration",
    "Enum",
    "Field",
    "Float32",
    "Float64",
    "FloatType",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Int128",
    "IntegerType",
    "List",
    "NestedType",
    "NumericType",
    "Object",
    "SignedIntegerType",
    "String",
    "Struct",
    "Time",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "UInt128",
    "Unknown",
    "UnsignedIntegerType",
]
