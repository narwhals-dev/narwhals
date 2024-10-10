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


class Int64(NumericType):
    """64-bit signed integer type.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> data = [2, 1, 3, 7]
        >>> ser_pd = pd.Series(data)
        >>> ser_pl = pl.Series(data)
        >>> ser_pa = pa.chunked_array([data])

        >>> nw.from_native(ser_pd, series_only=True).dtype
        Int64
        >>> nw.from_native(ser_pl, series_only=True).dtype
        Int64
        >>> nw.from_native(ser_pa, series_only=True).dtype
        Int64
    """


class Int32(NumericType):
    """32-bit signed integer type.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> data = [2, 1, 3, 7]
        >>> ser_pd = pd.Series(data)
        >>> ser_pl = pl.Series(data)
        >>> ser_pa = pa.chunked_array([data])

        >>> def func(ser):
        ...     ser_nw = nw.from_native(ser, series_only=True)
        ...     return ser_nw.cast(nw.Int32).dtype

        >>> func(ser_pd)
        Int32
        >>> func(ser_pl)
        Int32
        >>> func(ser_pa)
        Int32
    """


class Int16(NumericType):
    """16- bit signed integer type.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> data = [2, 1, 3, 7]
        >>> ser_pd = pd.Series(data)
        >>> ser_pl = pl.Series(data)
        >>> ser_pa = pa.chunked_array([data])

        >>> def func(ser):
        ...     ser_nw = nw.from_native(ser, series_only=True)
        ...     return ser_nw.cast(nw.Int16).dtype

        >>> func(ser_pd)
        Int16
        >>> func(ser_pl)
        Int16
        >>> func(ser_pa)
        Int16
    """


class Int8(NumericType):
    """8-bit signed integer type.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> data = [2, 1, 3, 7]
        >>> ser_pd = pd.Series(data)
        >>> ser_pl = pl.Series(data)
        >>> ser_pa = pa.chunked_array([data])

        >>> def func(ser):
        ...     ser_nw = nw.from_native(ser, series_only=True)
        ...     return ser_nw.cast(nw.Int8).dtype

        >>> func(ser_pd)
        Int8
        >>> func(ser_pl)
        Int8
        >>> func(ser_pa)
        Int8
    """


class UInt64(NumericType):
    """64-bit unsigned integer type.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> data = [2, 1, 3, 7]
        >>> ser_pd = pd.Series(data)
        >>> ser_pl = pl.Series(data)
        >>> ser_pa = pa.chunked_array([data])

        >>> def func(ser):
        ...     ser_nw = nw.from_native(ser, series_only=True)
        ...     return ser_nw.cast(nw.UInt64).dtype

        >>> func(ser_pd)
        UInt64
        >>> func(ser_pl)
        UInt64
        >>> func(ser_pa)
        UInt64
    """


class UInt32(NumericType):
    """32-bit unsigned integer type.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> data = [2, 1, 3, 7]
        >>> ser_pd = pd.Series(data)
        >>> ser_pl = pl.Series(data)
        >>> ser_pa = pa.chunked_array([data])

        >>> def func(ser):
        ...     ser_nw = nw.from_native(ser, series_only=True)
        ...     return ser_nw.cast(nw.UInt32).dtype

        >>> func(ser_pd)
        UInt32
        >>> func(ser_pl)
        UInt32
        >>> func(ser_pa)
        UInt32
    """


class UInt16(NumericType):
    """16-bit unsigned integer type.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> data = [2, 1, 3, 7]
        >>> ser_pd = pd.Series(data)
        >>> ser_pl = pl.Series(data)
        >>> ser_pa = pa.chunked_array([data])

        >>> def func(ser):
        ...     ser_nw = nw.from_native(ser, series_only=True)
        ...     return ser_nw.cast(nw.UInt16).dtype

        >>> func(ser_pd)
        UInt16
        >>> func(ser_pl)
        UInt16
        >>> func(ser_pa)
        UInt16
    """


class UInt8(NumericType):
    """8-bit unsigned integer type.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> data = [2, 1, 3, 7]
        >>> ser_pd = pd.Series(data)
        >>> ser_pl = pl.Series(data)
        >>> ser_pa = pa.chunked_array([data])

        >>> def func(ser):
        ...     ser_nw = nw.from_native(ser, series_only=True)
        ...     return ser_nw.cast(nw.UInt8).dtype

        >>> func(ser_pd)
        UInt8
        >>> func(ser_pl)
        UInt8
        >>> func(ser_pa)
        UInt8
    """


class Float64(NumericType):
    """64-bit floating point type.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> data = [0.001, 0.1, 0.01, 0.1]
        >>> ser_pd = pd.Series(data)
        >>> ser_pl = pl.Series(data)
        >>> ser_pa = pa.chunked_array([data])

        >>> nw.from_native(ser_pd, series_only=True).dtype
        Float64
        >>> nw.from_native(ser_pl, series_only=True).dtype
        Float64
        >>> nw.from_native(ser_pa, series_only=True).dtype
        Float64
    """


class Float32(NumericType):
    """32-bit floating point type.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> data = [0.001, 0.1, 0.01, 0.1]
        >>> ser_pd = pd.Series(data)
        >>> ser_pl = pl.Series(data)
        >>> ser_pa = pa.chunked_array([data])

        >>> def func(ser):
        ...     ser_nw = nw.from_native(ser, series_only=True)
        ...     return ser_nw.cast(nw.Float32).dtype

        >>> func(ser_pd)
        Float32
        >>> func(ser_pl)
        Float32
        >>> func(ser_pa)
        Float32
    """


class String(DType):
    """UTF-8 encoded string type.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> data = ["beluga", "narwhal", "orca", "vaquita"]
        >>> ser_pd = pd.Series(data)
        >>> ser_pl = pl.Series(data)
        >>> ser_pa = pa.chunked_array([data])

        >>> nw.from_native(ser_pd, series_only=True).dtype
        String
        >>> nw.from_native(ser_pl, series_only=True).dtype
        String
        >>> nw.from_native(ser_pa, series_only=True).dtype
        String
    """


class Boolean(DType):
    """Boolean type.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> data = [True, False, False, True]
        >>> ser_pd = pd.Series(data)
        >>> ser_pl = pl.Series(data)
        >>> ser_pa = pa.chunked_array([data])

        >>> nw.from_native(ser_pd, series_only=True).dtype
        Boolean
        >>> nw.from_native(ser_pl, series_only=True).dtype
        Boolean
        >>> nw.from_native(ser_pa, series_only=True).dtype
        Boolean
    """


class Object(DType): ...


class Unknown(DType):
    """Type representing DataType values that could not be determined statically.

    Examples:
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> data = [None, None, None, None]
        >>> ser_pl = pl.Series(data)
        >>> ser_pa = pa.chunked_array([data])

        >>> nw.from_native(ser_pl, series_only=True).dtype
        Unknown
        >>> nw.from_native(ser_pa, series_only=True).dtype
        Unknown
    """


class Datetime(TemporalType):
    """
    Data type representing a calendar date and time of day.

    Arguments:
        time_unit: Unit of time. Defaults to `'us'` (microseconds).
        time_zone: Time zone string, as defined in zoneinfo (to see valid strings run
            `import zoneinfo; zoneinfo.available_timezones()` for a full list).

    Notes:
        Adapted from Polars implementation at:
        https://github.com/pola-rs/polars/blob/py-1.7.1/py-polars/polars/datatypes/classes.py#L398-L457
    """

    def __init__(
        self: Self,
        time_unit: Literal["us", "ns", "ms", "s"] = "us",
        time_zone: str | timezone | None = None,
    ) -> None:
        if time_unit not in {"s", "ms", "us", "ns"}:
            msg = (
                "invalid `time_unit`"
                f"\n\nExpected one of {{'ns','us','ms', 's'}}, got {time_unit!r}."
            )
            raise ValueError(msg)

        if isinstance(time_zone, timezone):
            time_zone = str(time_zone)

        self.time_unit = time_unit
        self.time_zone = time_zone

    def __eq__(self: Self, other: object) -> bool:
        # allow comparing object instances to class
        if type(other) is type and issubclass(other, self.__class__):
            return True
        elif isinstance(other, self.__class__):
            return self.time_unit == other.time_unit and self.time_zone == other.time_zone
        else:  # pragma: no cover
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
        time_unit: Literal["us", "ns", "ms", "s"] = "us",
    ) -> None:
        if time_unit not in ("s", "ms", "us", "ns"):
            msg = (
                "invalid `time_unit`"
                f"\n\nExpected one of {{'ns','us','ms', 's'}}, got {time_unit!r}."
            )
            raise ValueError(msg)

        self.time_unit = time_unit

    def __eq__(self: Self, other: object) -> bool:
        # allow comparing object instances to class
        if type(other) is type and issubclass(other, self.__class__):
            return True
        elif isinstance(other, self.__class__):
            return self.time_unit == other.time_unit
        else:  # pragma: no cover
            return False

    def __hash__(self: Self) -> int:  # pragma: no cover
        return hash((self.__class__, self.time_unit))

    def __repr__(self: Self) -> str:  # pragma: no cover
        class_name = self.__class__.__name__
        return f"{class_name}(time_unit={self.time_unit!r})"


class Categorical(DType):
    """A categorical encoding of a set of strings.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> data = ["beluga", "narwhal", "orca", "vaquita"]
        >>> ser_pd = pd.Series(data)
        >>> ser_pl = pl.Series(data)
        >>> ser_pa = pa.chunked_array([data])

        >>> nw.from_native(ser_pd, series_only=True).cast(nw.Categorical).dtype
        Categorical
        >>> nw.from_native(ser_pl, series_only=True).cast(nw.Categorical).dtype
        Categorical
        >>> nw.from_native(ser_pa, series_only=True).cast(nw.Categorical).dtype
        Categorical
    """


class Enum(DType): ...


class Struct(DType):
    """Struct composite type.

    Examples:
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> data = [{"a": 1, "b": ["narwhal", "beluga"]}, {"a": 2, "b": ["orca"]}]
        >>> ser_pl = pl.Series(data)
        >>> ser_pa = pa.chunked_array([data])

        >>> nw.from_native(ser_pl, series_only=True).dtype
        Struct
        >>> nw.from_native(ser_pa, series_only=True).dtype
        Struct
    """


class List(DType):
    """Variable length list type.

    Examples:
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> data = [["narwhal", "orca"], ["beluga", "vaquita"]]
        >>> ser_pl = pl.Series(data)
        >>> ser_pa = pa.chunked_array([data])

        >>> nw.from_native(ser_pl, series_only=True).dtype
        List(String)
        >>> nw.from_native(ser_pa, series_only=True).dtype
        List(String)
    """

    def __init__(self, inner: DType | type[DType]) -> None:
        self.inner = inner

    def __eq__(self, other: DType | type[DType]) -> bool:  # type: ignore[override]
        # This equality check allows comparison of type classes and type instances.
        # If a parent type is not specific about its inner type, we infer it as equal:
        # > list[i64] == list[i64] -> True
        # > list[i64] == list[f32] -> False
        # > list[i64] == list      -> True

        # allow comparing object instances to class
        if type(other) is type and issubclass(other, self.__class__):
            return True
        elif isinstance(other, self.__class__):
            return self.inner == other.inner
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.__class__, self.inner))

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}({self.inner!r})"


class Array(DType):
    def __init__(self, inner: DType | type[DType], width: int | None = None) -> None:
        self.inner = inner
        if width is None:
            error = "`width` must be specified when initializing an `Array`"
            raise TypeError(error)
        self.width = width

    def __eq__(self, other: DType | type[DType]) -> bool:  # type: ignore[override]
        # This equality check allows comparison of type classes and type instances.
        # If a parent type is not specific about its inner type, we infer it as equal:
        # > array[i64] == array[i64] -> True
        # > array[i64] == array[f32] -> False
        # > array[i64] == array      -> True

        # allow comparing object instances to class
        if type(other) is type and issubclass(other, self.__class__):
            return True
        elif isinstance(other, self.__class__):
            return self.inner == other.inner
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.__class__, self.inner, self.width))

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}({self.inner!r}, {self.width})"


class Date(TemporalType): ...
