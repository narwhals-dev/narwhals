from __future__ import annotations

import enum
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from datetime import timezone
from functools import lru_cache
from itertools import starmap
from typing import TYPE_CHECKING

from narwhals._utils import (
    _DeferredIterable,
    isinstance_or_issubclass,
    qualified_type_name,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from typing import Any

    import _typeshed
    from typing_extensions import Self, TypeIs

    from narwhals.typing import DTypes, IntoDType, TimeUnit


def _validate_dtype(dtype: DType | type[DType]) -> None:
    if not isinstance_or_issubclass(dtype, DType):
        msg = (
            f"Expected Narwhals dtype, got: {type(dtype)}.\n\n"
            "Hint: if you were trying to cast to a type, use e.g. nw.Int64 instead of 'int64'."
        )
        raise TypeError(msg)


def _is_into_dtype(obj: Any) -> TypeIs[IntoDType]:
    return isinstance(obj, DType) or (
        isinstance(obj, DTypeClass) and not issubclass(obj, NestedType)
    )


def _is_nested_type(obj: Any) -> TypeIs[type[NestedType]]:
    return isinstance(obj, DTypeClass) and issubclass(obj, NestedType)


def _validate_into_dtype(dtype: Any) -> None:
    if not _is_into_dtype(dtype):
        if _is_nested_type(dtype):
            name = f"nw.{dtype.__name__}"
            msg = (
                f"{name!r} is not valid in this context.\n\n"
                f"Hint: instead of:\n\n"
                f"    {name}\n\n"
                "use:\n\n"
                f"    {name}(...)"
            )
        else:
            msg = f"Expected Narwhals dtype, got: {qualified_type_name(dtype)!r}."
        raise TypeError(msg)


class DTypeClass(type):
    """Metaclass for DType classes.

    - Nicely print classes.
    - Ensure [`__slots__`] are always defined to prevent `__dict__` creation (empty by default).

    [`__slots__`]: https://docs.python.org/3/reference/datamodel.html#object.__slots__
    """

    def __repr__(cls) -> str:
        return cls.__name__

    # https://github.com/python/typeshed/blob/776508741d76b58f9dcb2aaf42f7d4596a48d580/stdlib/abc.pyi#L13-L19
    # https://github.com/python/typeshed/blob/776508741d76b58f9dcb2aaf42f7d4596a48d580/stdlib/_typeshed/__init__.pyi#L36-L40
    # https://github.com/astral-sh/ruff/issues/8353#issuecomment-1786238311
    # https://docs.python.org/3/reference/datamodel.html#creating-the-class-object
    def __new__(
        metacls: type[_typeshed.Self],
        cls_name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        **kwds: Any,
    ) -> _typeshed.Self:
        namespace.setdefault("__slots__", ())
        return super().__new__(metacls, cls_name, bases, namespace, **kwds)  # type: ignore[no-any-return, misc]


class DType(metaclass=DTypeClass):
    """Base class for all Narwhals data types."""

    __slots__ = ()  # NOTE: Keep this one defined manually for the type checker

    def __repr__(self) -> str:
        return self.__class__.__qualname__

    @classmethod
    def base_type(cls) -> type[Self]:
        """Return this DType's fundamental/root type class.

        Examples:
            >>> import narwhals as nw
            >>> nw.Datetime("us").base_type()
            Datetime
            >>> nw.String.base_type()
            String
            >>> nw.List(nw.Int64).base_type()
            List
        """
        return cls

    @classmethod
    def is_numeric(cls: type[Self]) -> bool:
        """Check whether the data type is a numeric type."""
        return issubclass(cls, NumericType)

    @classmethod
    def is_integer(cls: type[Self]) -> bool:
        """Check whether the data type is an integer type."""
        return issubclass(cls, IntegerType)

    @classmethod
    def is_signed_integer(cls: type[Self]) -> bool:
        """Check whether the data type is a signed integer type."""
        return issubclass(cls, SignedIntegerType)

    @classmethod
    def is_unsigned_integer(cls: type[Self]) -> bool:
        """Check whether the data type is an unsigned integer type."""
        return issubclass(cls, UnsignedIntegerType)

    @classmethod
    def is_float(cls: type[Self]) -> bool:
        """Check whether the data type is a floating point type."""
        return issubclass(cls, FloatType)

    @classmethod
    def is_decimal(cls: type[Self]) -> bool:
        """Check whether the data type is a decimal type."""
        return issubclass(cls, Decimal)

    @classmethod
    def is_temporal(cls: type[Self]) -> bool:
        """Check whether the data type is a temporal type."""
        return issubclass(cls, TemporalType)

    @classmethod
    def is_nested(cls: type[Self]) -> bool:
        """Check whether the data type is a nested type."""
        return issubclass(cls, NestedType)

    @classmethod
    def is_boolean(cls: type[Self]) -> bool:
        """Check whether the data type is a boolean type."""
        return issubclass(cls, Boolean)

    def __eq__(self, other: DType | type[DType]) -> bool:  # type: ignore[override]
        """Check if this DType is equivalent to another DType.

        Examples:
            >>> import narwhals as nw
            >>> nw.String() == nw.String()
            True
            >>> nw.String() == nw.String
            True
            >>> nw.Int16() == nw.Int32
            False
            >>> nw.Boolean() == nw.Int8
            False
            >>> nw.Date() == nw.Datetime
            False
        """
        return isinstance_or_issubclass(other, type(self))

    def __hash__(self) -> int:
        return hash(self.__class__)


class NumericType(DType):
    """Base class for numeric data types."""


class IntegerType(NumericType):
    """Base class for integer data types."""


class SignedIntegerType(IntegerType):
    """Base class for signed integer data types."""


class UnsignedIntegerType(IntegerType):
    """Base class for unsigned integer data types."""


class FloatType(NumericType):
    """Base class for float data types."""


class TemporalType(DType):
    """Base class for temporal data types."""


class NestedType(DType):
    """Base class for nested data types."""


class Decimal(NumericType):
    """Decimal type.

    Examples:
        >>> import polars as pl
        >>> import narwhals as nw
        >>> s = pl.Series(["1.5"], dtype=pl.Decimal)
        >>> nw.from_native(s, series_only=True).dtype
        Decimal
    """


class Int128(SignedIntegerType):
    """128-bit signed integer type.

    Examples:
        >>> import polars as pl
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> import duckdb
        >>> s_native = pl.Series([2, 1, 3, 7])
        >>> s = nw.from_native(s_native, series_only=True)
        >>> df_native = pa.table({"a": [2, 1, 3, 7]})
        >>> rel = duckdb.sql(" SELECT CAST (a AS INT128) AS a FROM df_native ")

        >>> s.cast(nw.Int128).dtype
        Int128
        >>> nw.from_native(rel).collect_schema()["a"]
        Int128
    """


class Int64(SignedIntegerType):
    """64-bit signed integer type.

    Examples:
        >>> import polars as pl
        >>> import narwhals as nw
        >>> s_native = pl.Series([2, 1, 3, 7])
        >>> s = nw.from_native(s_native, series_only=True)
        >>> s.cast(nw.Int64).dtype
        Int64
    """


class Int32(SignedIntegerType):
    """32-bit signed integer type.

    Examples:
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> s_native = pa.chunked_array([[2, 1, 3, 7]])
        >>> s = nw.from_native(s_native, series_only=True)
        >>> s.cast(nw.Int32).dtype
        Int32
    """


class Int16(SignedIntegerType):
    """16-bit signed integer type.

    Examples:
        >>> import polars as pl
        >>> import narwhals as nw
        >>> s_native = pl.Series([2, 1, 3, 7])
        >>> s = nw.from_native(s_native, series_only=True)
        >>> s.cast(nw.Int16).dtype
        Int16
    """


class Int8(SignedIntegerType):
    """8-bit signed integer type.

    Examples:
        >>> import pandas as pd
        >>> import narwhals as nw
        >>> s_native = pd.Series([2, 1, 3, 7])
        >>> s = nw.from_native(s_native, series_only=True)
        >>> s.cast(nw.Int8).dtype
        Int8
    """


class UInt128(UnsignedIntegerType):
    """128-bit unsigned integer type.

    Examples:
        >>> import pandas as pd
        >>> import narwhals as nw
        >>> import duckdb
        >>> df_native = pd.DataFrame({"a": [2, 1, 3, 7]})
        >>> rel = duckdb.sql(" SELECT CAST (a AS UINT128) AS a FROM df_native ")
        >>> nw.from_native(rel).collect_schema()["a"]
        UInt128
    """


class UInt64(UnsignedIntegerType):
    """64-bit unsigned integer type.

    Examples:
        >>> import pandas as pd
        >>> import narwhals as nw
        >>> s_native = pd.Series([2, 1, 3, 7])
        >>> s = nw.from_native(s_native, series_only=True)
        >>> s.cast(nw.UInt64).dtype
        UInt64
    """


class UInt32(UnsignedIntegerType):
    """32-bit unsigned integer type.

    Examples:
        >>> import polars as pl
        >>> import narwhals as nw
        >>> s_native = pl.Series([2, 1, 3, 7])
        >>> s = nw.from_native(s_native, series_only=True)
        >>> s.cast(nw.UInt32).dtype
        UInt32
    """


class UInt16(UnsignedIntegerType):
    """16-bit unsigned integer type.

    Examples:
        >>> import polars as pl
        >>> import narwhals as nw
        >>> s_native = pl.Series([2, 1, 3, 7])
        >>> s = nw.from_native(s_native, series_only=True)
        >>> s.cast(nw.UInt16).dtype
        UInt16
    """


class UInt8(UnsignedIntegerType):
    """8-bit unsigned integer type.

    Examples:
        >>> import polars as pl
        >>> import narwhals as nw
        >>> s_native = pl.Series([2, 1, 3, 7])
        >>> s = nw.from_native(s_native, series_only=True)
        >>> s.cast(nw.UInt8).dtype
        UInt8
    """


class Float64(FloatType):
    """64-bit floating point type.

    Examples:
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> s_native = pa.chunked_array([[0.001, 0.1, 0.01, 0.1]])
        >>> s = nw.from_native(s_native, series_only=True)
        >>> s.cast(nw.Float64).dtype
        Float64
    """


class Float32(FloatType):
    """32-bit floating point type.

    Examples:
        >>> import polars as pl
        >>> import narwhals as nw
        >>> s_native = pl.Series([0.001, 0.1, 0.01, 0.1])
        >>> s = nw.from_native(s_native, series_only=True)
        >>> s.cast(nw.Float32).dtype
        Float32
    """


class String(DType):
    """UTF-8 encoded string type.

    Examples:
        >>> import pandas as pd
        >>> import narwhals as nw
        >>> s_native = pd.Series(["beluga", "narwhal", "orca", "vaquita"])
        >>> nw.from_native(s_native, series_only=True).dtype
        String
    """


class Boolean(DType):
    """Boolean type.

    Examples:
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> s_native = pa.chunked_array([[True, False, False, True]])
        >>> nw.from_native(s_native, series_only=True).dtype
        Boolean
    """


class Object(DType):
    """Data type for wrapping arbitrary Python objects.

    Examples:
        >>> import pandas as pd
        >>> import narwhals as nw
        >>> class Foo: ...
        >>> s_native = pd.Series([Foo(), Foo()])
        >>> nw.from_native(s_native, series_only=True).dtype
        Object
    """


class Unknown(DType):
    """Type representing DataType values that could not be determined statically.

    Examples:
        >>> import pandas as pd
        >>> import narwhals as nw
        >>> s_native = pd.Series(pd.period_range("2000-01", periods=4, freq="M"))
        >>> nw.from_native(s_native, series_only=True).dtype
        Unknown
    """


class _DatetimeMeta(DTypeClass):
    @property
    def time_unit(cls) -> TimeUnit:
        """Unit of time. Defaults to `'us'` (microseconds)."""
        return "us"

    @property
    def time_zone(cls) -> str | None:
        """Time zone string. Defaults to `None`."""
        return None


class Datetime(TemporalType, metaclass=_DatetimeMeta):
    """Data type representing a calendar date and time of day.

    Arguments:
        time_unit: Unit of time. Defaults to `'us'` (microseconds).
        time_zone: Time zone string, as defined in zoneinfo (to see valid strings run
            `import zoneinfo; zoneinfo.available_timezones()` for a full list).

    Notes:
        Adapted from [Polars implementation](https://github.com/pola-rs/polars/blob/py-1.7.1/py-polars/polars/datatypes/classes.py#L398-L457)

    Examples:
        >>> from datetime import datetime, timedelta
        >>> import polars as pl
        >>> import narwhals as nw
        >>> s_native = (
        ...     pl.Series([datetime(2024, 12, 9) + timedelta(days=n) for n in range(5)])
        ...     .cast(pl.Datetime("ms"))
        ...     .dt.replace_time_zone("Africa/Accra")
        ... )
        >>> nw.from_native(s_native, series_only=True).dtype
        Datetime(time_unit='ms', time_zone='Africa/Accra')
    """

    __slots__ = ("time_unit", "time_zone")

    def __init__(
        self, time_unit: TimeUnit = "us", time_zone: str | timezone | None = None
    ) -> None:
        if time_unit not in {"s", "ms", "us", "ns"}:
            msg = (
                "invalid `time_unit`"
                f"\n\nExpected one of {{'ns','us','ms', 's'}}, got {time_unit!r}."
            )
            raise ValueError(msg)

        if isinstance(time_zone, timezone):
            time_zone = str(time_zone)

        self.time_unit: TimeUnit = time_unit
        """Unit of time."""
        self.time_zone: str | None = time_zone
        """Time zone string, as defined in zoneinfo.

        Notes:
            To see valid strings run `import zoneinfo; zoneinfo.available_timezones()` for a full list.
        """

    def __eq__(self, other: DType | type[DType]) -> bool:  # type: ignore[override]
        """Check if this Datetime is equivalent to another DType.

        Examples:
            >>> import narwhals as nw
            >>> nw.Datetime("s") == nw.Datetime("s")
            True
            >>> nw.Datetime() == nw.Datetime("us")
            True
            >>> nw.Datetime("us") == nw.Datetime("ns")
            False
            >>> nw.Datetime("us", "UTC") == nw.Datetime(time_unit="us", time_zone="UTC")
            True
            >>> nw.Datetime(time_zone="UTC") == nw.Datetime(time_zone="EST")
            False
            >>> nw.Datetime() == nw.Duration()
            False
            >>> nw.Datetime("ms") == nw.Datetime
            True
        """
        if type(other) is _DatetimeMeta:
            return True
        if isinstance(other, self.__class__):
            return self.time_unit == other.time_unit and self.time_zone == other.time_zone
        return False  # pragma: no cover

    def __hash__(self) -> int:  # pragma: no cover
        return hash((self.__class__, self.time_unit, self.time_zone))

    def __repr__(self) -> str:  # pragma: no cover
        class_name = self.__class__.__name__
        return f"{class_name}(time_unit={self.time_unit!r}, time_zone={self.time_zone!r})"


class _DurationMeta(DTypeClass):
    @property
    def time_unit(cls) -> TimeUnit:
        """Unit of time. Defaults to `'us'` (microseconds)."""
        return "us"


class Duration(TemporalType, metaclass=_DurationMeta):
    """Data type representing a time duration.

    Arguments:
        time_unit: Unit of time. Defaults to `'us'` (microseconds).

    Notes:
        Adapted from [Polars implementation](https://github.com/pola-rs/polars/blob/py-1.7.1/py-polars/polars/datatypes/classes.py#L460-L502)

    Examples:
        >>> from datetime import timedelta
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> s_native = pa.chunked_array(
        ...     [[timedelta(seconds=d) for d in range(1, 4)]], type=pa.duration("ms")
        ... )
        >>> nw.from_native(s_native, series_only=True).dtype
        Duration(time_unit='ms')
    """

    __slots__ = ("time_unit",)

    def __init__(self, time_unit: TimeUnit = "us") -> None:
        if time_unit not in {"s", "ms", "us", "ns"}:
            msg = (
                "invalid `time_unit`"
                f"\n\nExpected one of {{'ns','us','ms', 's'}}, got {time_unit!r}."
            )
            raise ValueError(msg)

        self.time_unit: TimeUnit = time_unit
        """Unit of time."""

    def __eq__(self, other: DType | type[DType]) -> bool:  # type: ignore[override]
        """Check if this Duration is equivalent to another DType.

        Examples:
            >>> import narwhals as nw
            >>> nw.Duration("us") == nw.Duration("us")
            True
            >>> nw.Duration() == nw.Duration("us")
            True
            >>> nw.Duration("us") == nw.Duration("ns")
            False
            >>> nw.Duration() == nw.Datetime()
            False
            >>> nw.Duration("ms") == nw.Duration
            True
        """
        if type(other) is _DurationMeta:
            return True
        if isinstance(other, self.__class__):
            return self.time_unit == other.time_unit
        return False  # pragma: no cover

    def __hash__(self) -> int:  # pragma: no cover
        return hash((self.__class__, self.time_unit))

    def __repr__(self) -> str:  # pragma: no cover
        class_name = self.__class__.__name__
        return f"{class_name}(time_unit={self.time_unit!r})"


class Categorical(DType):
    """A categorical encoding of a set of strings.

    Examples:
        >>> import polars as pl
        >>> import narwhals as nw
        >>> s_native = pl.Series(["beluga", "narwhal", "orca"])
        >>> nw.from_native(s_native, series_only=True).cast(nw.Categorical).dtype
        Categorical
    """


class Enum(DType):
    """A fixed categorical encoding of a unique set of strings.

    Polars has an Enum data type. In pandas, ordered categories get mapped
    to Enum. PyArrow has no Enum equivalent.

    Examples:
       >>> import narwhals as nw
       >>> nw.Enum(["beluga", "narwhal", "orca"])
       Enum(categories=['beluga', 'narwhal', 'orca'])
    """

    __slots__ = ("_cached_categories", "_delayed_categories")

    def __init__(self, categories: Iterable[str] | type[enum.Enum]) -> None:
        self._delayed_categories: _DeferredIterable[str] | None = None
        self._cached_categories: tuple[str, ...] | None = None

        if isinstance(categories, _DeferredIterable):
            self._delayed_categories = categories
        elif isinstance(categories, type) and issubclass(categories, enum.Enum):
            self._cached_categories = tuple(member.value for member in categories)
        else:
            self._cached_categories = tuple(categories)

    @property
    def categories(self) -> tuple[str, ...]:
        """The categories in the dataset."""
        if (cached := self._cached_categories) is not None:
            return cached
        if (delayed := self._delayed_categories) is not None:
            self._cached_categories = delayed.to_tuple()
            return self._cached_categories
        msg = f"Internal structure of {type(self).__name__!r} is invalid."  # pragma: no cover
        raise TypeError(msg)  # pragma: no cover

    def __eq__(self, other: DType | type[DType]) -> bool:  # type: ignore[override]
        """Check if this Enum is equivalent to another DType.

        Examples:
            >>> import narwhals as nw
            >>> nw.Enum(["a", "b", "c"]) == nw.Enum(["a", "b", "c"])
            True
            >>> import polars as pl
            >>> categories = pl.Series(["a", "b", "c"])
            >>> nw.Enum(["a", "b", "c"]) == nw.Enum(categories)
            True
            >>> nw.Enum(["a", "b", "c"]) == nw.Enum(["b", "a", "c"])
            False
            >>> nw.Enum(["a", "b", "c"]) == nw.Enum(["a"])
            False
            >>> nw.Enum(["a", "b", "c"]) == nw.Categorical
            False
            >>> nw.Enum(["a", "b", "c"]) == nw.Enum
            True
        """
        if type(other) is DTypeClass:
            return other is Enum
        return isinstance(other, type(self)) and self.categories == other.categories

    def __hash__(self) -> int:
        return hash((self.__class__, tuple(self.categories)))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(categories={list(self.categories)!r})"


class Field:
    """Definition of a single field within a `Struct` DType.

    Arguments:
        name: The name of the field within its parent `Struct`.
        dtype: The `DType` of the field's values.

    Examples:
       >>> import pyarrow as pa
       >>> import narwhals as nw
       >>> data = [{"a": 1, "b": ["narwhal", "beluga"]}, {"a": 2, "b": ["orca"]}]
       >>> ser_pa = pa.chunked_array([data])
       >>> nw.from_native(ser_pa, series_only=True).dtype.fields
       [Field('a', Int64), Field('b', List(String))]
    """

    __slots__ = ("dtype", "name")
    name: str
    """The name of the field within its parent `Struct`."""
    dtype: IntoDType
    """The `DType` of the field's values."""

    def __init__(self, name: str, dtype: IntoDType) -> None:
        self.name = name
        self.dtype = dtype

    def __eq__(self, other: Field) -> bool:  # type: ignore[override]
        """Check if this Field is equivalent to another Field.

        Two fields are equivalent if they have the same name and the same dtype.

        Examples:
            >>> import narwhals as nw
            >>> nw.Field("a", nw.String) == nw.Field("a", nw.String())
            True
            >>> nw.Field("a", nw.String) == nw.Field("a", nw.String)
            True
            >>> nw.Field("a", nw.String) == nw.Field("a", nw.Datetime)
            False
            >>> nw.Field("a", nw.String) == nw.Field("b", nw.String)
            False
            >>> nw.Field("a", nw.String) == nw.String
            False
        """
        return (
            isinstance(other, Field)
            and (self.name == other.name)
            and (self.dtype == other.dtype)
        )

    def __hash__(self) -> int:
        return hash((self.name, self.dtype))

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}({self.name!r}, {self.dtype})"


class Struct(NestedType):
    """Struct composite type.

    Arguments:
        fields: The fields that make up the struct. Can be either a sequence of Field
            objects or a mapping of column names to data types.

    Examples:
       >>> import pyarrow as pa
       >>> import narwhals as nw
       >>> s_native = pa.chunked_array(
       ...     [[{"a": 1, "b": ["narwhal", "beluga"]}, {"a": 2, "b": ["orca"]}]]
       ... )
       >>> nw.from_native(s_native, series_only=True).dtype
       Struct({'a': Int64, 'b': List(String)})
    """

    __slots__ = ("fields",)
    fields: list[Field]
    """The fields that make up the struct."""

    def __init__(self, fields: Sequence[Field] | Mapping[str, IntoDType]) -> None:
        if isinstance(fields, Mapping):
            self.fields = list(starmap(Field, fields.items()))
        else:
            self.fields = list(fields)

    def __eq__(self, other: DType | type[DType]) -> bool:  # type: ignore[override]
        """Check if this Struct is equivalent to another DType.

        Examples:
            >>> import narwhals as nw
            >>> nw.Struct({"a": nw.Int64}) == nw.Struct({"a": nw.Int64})
            True
            >>> nw.Struct({"a": nw.Int64}) == nw.Struct({"a": nw.Boolean})
            False
            >>> nw.Struct({"a": nw.Int64}) == nw.Struct({"b": nw.Int64})
            False
            >>> nw.Struct({"a": nw.Int64}) == nw.Struct([nw.Field("a", nw.Int64)])
            True

            If a parent type is not specific about its inner type, we infer it as equal

            >>> nw.Struct({"a": nw.Int64}) == nw.Struct
            True
        """
        if type(other) is DTypeClass and issubclass(other, self.__class__):
            return True
        if isinstance(other, self.__class__):
            return self.fields == other.fields
        return False

    def __hash__(self) -> int:
        return hash((self.__class__, tuple(self.fields)))

    def __iter__(self) -> Iterator[tuple[str, IntoDType]]:  # pragma: no cover
        for fld in self.fields:
            yield fld.name, fld.dtype

    def __reversed__(self) -> Iterator[tuple[str, IntoDType]]:
        for fld in reversed(self.fields):
            yield fld.name, fld.dtype

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}({dict(self)})"

    def to_schema(self) -> OrderedDict[str, IntoDType]:
        """Return Struct dtype as a schema dict."""
        return OrderedDict(self)


class List(NestedType):
    """Variable length list type.

    Examples:
       >>> import pandas as pd
       >>> import pyarrow as pa
       >>> import narwhals as nw
       >>> s_native = pd.Series(
       ...     [["narwhal", "orca"], ["beluga", "vaquita"]],
       ...     dtype=pd.ArrowDtype(pa.large_list(pa.large_string())),
       ... )
       >>> nw.from_native(s_native, series_only=True).dtype
       List(String)
    """

    __slots__ = ("inner",)
    inner: IntoDType
    """The DType of the values within each list."""

    def __init__(self, inner: IntoDType) -> None:
        self.inner = inner

    def __eq__(self, other: DType | type[DType]) -> bool:  # type: ignore[override]
        """Check if this List is equivalent to another DType.

        Examples:
            >>> import narwhals as nw
            >>> nw.List(nw.Int64) == nw.List(nw.Int64)
            True
            >>> nw.List(nw.Int64) == nw.List(nw.Float32)
            False

            If a parent type is not specific about its inner type, we infer it as equal

            >>> nw.List(nw.Int64) == nw.List
            True
        """
        if type(other) is DTypeClass and issubclass(other, self.__class__):
            return True
        if isinstance(other, self.__class__):
            return self.inner == other.inner
        return False

    def __hash__(self) -> int:
        return hash((self.__class__, self.inner))

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}({self.inner!r})"


class Array(NestedType):
    """Fixed length list type.

    Arguments:
        inner: The datatype of the values within each array.
        shape: The shape of the arrays.

    Examples:
        >>> import polars as pl
        >>> import narwhals as nw
        >>> s_native = pl.Series([[1, 2], [3, 4], [5, 6]], dtype=pl.Array(pl.Int32, 2))
        >>> nw.from_native(s_native, series_only=True).dtype
        Array(Int32, shape=(2,))
    """

    __slots__ = ("inner", "shape", "size")
    inner: IntoDType
    """The DType of the values within each array."""
    size: int
    """The size of the Array."""
    shape: tuple[int, ...]
    """The shape of the arrays."""

    def __init__(self, inner: IntoDType, shape: int | tuple[int, ...]) -> None:
        inner_shape: tuple[int, ...] = inner.shape if isinstance(inner, Array) else ()
        if isinstance(shape, int):
            self.inner = inner
            self.size = shape
            self.shape = (shape, *inner_shape)

        elif isinstance(shape, tuple) and len(shape) != 0 and isinstance(shape[0], int):
            if len(shape) > 1:
                inner = Array(inner, shape[1:])

            self.inner = inner
            self.size = shape[0]
            self.shape = shape + inner_shape

        else:
            msg = f"invalid input for shape: {shape!r}"
            raise TypeError(msg)

    def __eq__(self, other: DType | type[DType]) -> bool:  # type: ignore[override]
        """Check if this Array is equivalent to another DType.

        Examples:
            >>> import narwhals as nw
            >>> nw.Array(nw.Int64, 2) == nw.Array(nw.Int64, 2)
            True
            >>> nw.Array(nw.Int64, 2) == nw.Array(nw.String, 2)
            False
            >>> nw.Array(nw.Int64, 2) == nw.Array(nw.Int64, 4)
            False

            If a parent type is not specific about its inner type, we infer it as equal

            >>> nw.Array(nw.Int64, 2) == nw.Array
            True
        """
        if type(other) is DTypeClass and issubclass(other, self.__class__):
            return True
        if isinstance(other, self.__class__):
            if self.shape != other.shape:
                return False
            return self.inner == other.inner
        return False

    def __hash__(self) -> int:
        return hash((self.__class__, self.inner, self.shape))

    def __repr__(self) -> str:
        # Get leaf type
        dtype_ = self
        for _ in self.shape:
            dtype_ = dtype_.inner  # type: ignore[assignment]

        class_name = self.__class__.__name__
        return f"{class_name}({dtype_!r}, shape={self.shape})"


class Date(TemporalType):
    """Data type representing a calendar date.

    Examples:
        >>> from datetime import date, timedelta
        >>> import pyarrow as pa
        >>> import narwhals as nw
        >>> s_native = pa.chunked_array(
        ...     [[date(2024, 12, 1) + timedelta(days=d) for d in range(4)]]
        ... )
        >>> nw.from_native(s_native, series_only=True).dtype
        Date
    """


class Time(TemporalType):
    """Data type representing the time of day.

    Examples:
       >>> import polars as pl
       >>> import pyarrow as pa
       >>> import narwhals as nw
       >>> import duckdb
       >>> from datetime import time
       >>> data = [time(9, 0), time(9, 1, 10), time(9, 2)]
       >>> ser_pl = pl.Series(data)
       >>> ser_pa = pa.chunked_array([pa.array(data, type=pa.time64("ns"))])
       >>> rel = duckdb.sql(
       ...     " SELECT * FROM (VALUES (TIME '12:00:00'), (TIME '14:30:15')) df(t)"
       ... )

       >>> nw.from_native(ser_pl, series_only=True).dtype
       Time
       >>> nw.from_native(ser_pa, series_only=True).dtype
       Time
       >>> nw.from_native(rel).collect_schema()["t"]
       Time
    """


class Binary(DType):
    """Binary type.

    Examples:
        >>> import polars as pl
        >>> import narwhals as nw
        >>> import pyarrow as pa
        >>> import duckdb
        >>> data = [b"test1", b"test2"]
        >>> ser_pl = pl.Series(data, dtype=pl.Binary)
        >>> ser_pa = pa.chunked_array([pa.array(data, type=pa.binary())])
        >>> rel = duckdb.sql(
        ...     "SELECT * FROM (VALUES (BLOB 'test1'), (BLOB 'test2')) AS df(t)"
        ... )

        >>> nw.from_native(ser_pl, series_only=True).dtype
        Binary
        >>> nw.from_native(ser_pa, series_only=True).dtype
        Binary
        >>> nw.from_native(rel).collect_schema()["t"]
        Binary
    """


@lru_cache(maxsize=4)
def _time_unit_to_index(time_unit: TimeUnit) -> int:
    """Convert time unit to an index for comparison (larger = more precise)."""
    return {"s": 0, "ms": 1, "us": 2, "ns": 3}[time_unit]


def _min_time_unit(a: TimeUnit, b: TimeUnit) -> TimeUnit:
    """Return the less precise time unit."""
    return a if _time_unit_to_index(a) <= _time_unit_to_index(b) else b


@lru_cache(4)
def _signed_int_to_bit_size(*, dtypes: DTypes) -> dict[DType, int]:
    """Mapping from signed integer types to their bit size"""
    return {
        dtypes.Int8(): 8,
        dtypes.Int16(): 16,
        dtypes.Int32(): 32,
        dtypes.Int64(): 64,
        dtypes.Int128(): 128,
    }


@lru_cache(4)
def _unsigned_int_to_bit_size(*, dtypes: DTypes) -> dict[DType, int]:
    """Mapping from bit size to signed integer type"""
    return {
        dtypes.UInt8(): 8,
        dtypes.UInt16(): 16,
        dtypes.UInt32(): 32,
        dtypes.UInt64(): 64,
        dtypes.UInt128(): 128,
    }


@lru_cache(4)
def _bit_size_to_signed_int(*, dtypes: DTypes) -> dict[int, DType]:
    """Mapping from bit size to signed integer type"""
    return {v: k for k, v in _signed_int_to_bit_size(dtypes=dtypes).items()}


@lru_cache(4)
def _bit_size_to_unsigned_int(*, dtypes: DTypes) -> dict[int, DType]:
    """Mapping from bit size to unsigned integer type"""
    return {v: k for k, v in _unsigned_int_to_bit_size(dtypes=dtypes).items()}


def _get_integer_supertype(left: DType, right: DType, *, dtypes: DTypes) -> DType | None:
    """Get supertype for two integer types.

    Following Polars rules:

    - Same signedness: return the larger type
    - Mixed signedness: promote to signed with enough bits to hold both
    - Int64 + UInt64 -> Float64 (following Polars)
    """
    left_signed = left.is_signed_integer()
    right_signed = right.is_signed_integer()

    signed_int_to_bit_size = _signed_int_to_bit_size(dtypes=dtypes)
    unsigned_int_to_bit_size = _unsigned_int_to_bit_size(dtypes=dtypes)

    left_bits = signed_int_to_bit_size.get(left) or unsigned_int_to_bit_size.get(left)
    right_bits = signed_int_to_bit_size.get(right) or unsigned_int_to_bit_size.get(right)

    if left_bits is None or right_bits is None:  # pragma: no cover
        return None

    # Same signedness: return larger type
    if left_signed == right_signed:
        max_bits = max(left_bits, right_bits)
        return (
            _bit_size_to_signed_int(dtypes=dtypes)[max_bits]
            if left_signed
            else _bit_size_to_unsigned_int(dtypes=dtypes)[max_bits]
        )

    # Mixed signedness: need signed type that can hold both
    # The unsigned type needs to fit in a signed type with more bits
    signed_bits, unsigned_bits = (
        (left_bits, right_bits) if left_signed else (right_bits, left_bits)
    )

    # If signed type is strictly larger than unsigned, it can hold both
    if signed_bits > unsigned_bits:
        return _bit_size_to_signed_int(dtypes=dtypes)[signed_bits]

    # Otherwise, need to go to the next larger signed type
    # For Int64 + UInt64, Polars uses Float64 instead of Int128
    if unsigned_bits >= 64:
        return dtypes.Float64()

    # Find the smallest signed integer that can hold the unsigned value
    required_bits = unsigned_bits * 2
    for bits in (16, 32, 64):
        if bits >= required_bits:
            return _bit_size_to_signed_int(dtypes=dtypes)[bits]

    # Fallback to Float64 if no integer type large enough
    return dtypes.Float64()


def get_supertype(left: DType, right: DType, *, dtypes: DTypes) -> DType | None:
    """Given two data types, determine the data type that both types can reasonably safely be cast to.

    This function follows Polars' supertype rules:
    https://github.com/pola-rs/polars/blob/main/crates/polars-core/src/utils/supertype.rs

    Arguments:
        left: First data type.
        right: Second data type.

    Returns:
        The common supertype that both types can be safely cast to, or None if no such type exists.

    Examples:
        >>> import narwhals as nw
        >>> from narwhals.dtypes import get_supertype
        >>> get_supertype(nw.Int32(), nw.Int64())
        Int64
        >>> get_supertype(nw.Int32(), nw.Float64())
        Float64
        >>> get_supertype(nw.UInt8(), nw.Int8())
        Int16
        >>> get_supertype(nw.Date(), nw.Datetime("us"))
        Datetime(time_unit='us', time_zone=None)
        >>> get_supertype(nw.String(), nw.Int64()) is None
        True
    """
    if isinstance(left, dtypes.Datetime) and isinstance(right, dtypes.Datetime):
        if left.time_zone != right.time_zone:
            return None
        return dtypes.Datetime(
            _min_time_unit(left.time_unit, right.time_unit), left.time_zone
        )

    if isinstance(left, dtypes.Duration) and isinstance(right, dtypes.Duration):
        return dtypes.Duration(_min_time_unit(left.time_unit, right.time_unit))

    # For Enum types, categories must match
    if isinstance(left, dtypes.Enum) and isinstance(right, dtypes.Enum):
        if (left_cats := left.categories) == (right_cats := right.categories):
            return left
        # TODO(FBruzzesi): Should we merge the categories? return dtypes.Enum((*left_cats, *right_cat))
        return dtypes.String()

    if isinstance(left, dtypes.List) and isinstance(right, dtypes.List):
        left_inner, right_inner = left.inner, right.inner
        # Handle case where inner is a type vs instance
        if isinstance(left_inner, type):
            left_inner = left_inner()
        if isinstance(right_inner, type):
            right_inner = right_inner()
        if (
            inner_super_type := get_supertype(left_inner, right_inner, dtypes=dtypes)
        ) is None:
            return None
        return List(inner_super_type)

    if isinstance(left, Array) and isinstance(right, Array):
        if left.shape != right.shape:
            return None
        left_inner, right_inner = left.inner, right.inner
        if isinstance(left_inner, type):
            left_inner = left_inner()
        if isinstance(right_inner, type):
            right_inner = right_inner()
        if (
            inner_super_type := get_supertype(left_inner, right_inner, dtypes=dtypes)
        ) is None:
            return None
        return Array(inner_super_type, left.size)

    if isinstance(left, dtypes.Struct) and isinstance(right, dtypes.Struct):
        # left_fields, right_fields = left.fields, right.fields
        msg = "TODO"
        raise NotImplementedError(msg)

    if left == right:
        return left

    # Numeric and Boolean -> Numeric
    if right.is_numeric() and left.is_boolean():
        return right
    if left.is_numeric() and right.is_boolean():
        return left

    # Both Integer
    if left.is_integer() and right.is_integer():
        return _get_integer_supertype(left, right, dtypes=dtypes)

    # Both Float
    if left.is_float() and right.is_float():
        return (
            dtypes.Float64()
            if (left == dtypes.Float64() or right == dtypes.Float64())
            else dtypes.Float32()
        )

    # Integer + Float -> Float
    #  * Small integers (Int8, Int16, UInt8, UInt16) + Float32 -> Float32
    #  * Larger integers (Int32+) + Float32 -> Float64
    #  * Any integer + Float64 -> Float64
    if left.is_integer() and right.is_float():
        if right == dtypes.Float64():
            return dtypes.Float64()

        # Float32 case
        left_bits = _signed_int_to_bit_size(dtypes=dtypes).get(
            left
        ) or _unsigned_int_to_bit_size(dtypes=dtypes).get(left)
        if left_bits is not None and left_bits <= 16:
            return dtypes.Float32()
        return dtypes.Float64()

    if right.is_integer() and left.is_float():
        if left == dtypes.Float64():
            return dtypes.Float64()

        # Float32 case
        right_bits = _signed_int_to_bit_size(dtypes=dtypes).get(
            right
        ) or _unsigned_int_to_bit_size(dtypes=dtypes).get(right)
        if right_bits is not None and right_bits <= 16:
            return dtypes.Float32()
        return dtypes.Float64()

    # Decimal with other numeric types
    if (isinstance(left, dtypes.Decimal) and right.is_numeric()) or (
        isinstance(right, dtypes.Decimal) and left.is_numeric()
    ):
        return dtypes.Decimal()

    # Date + Datetime -> Datetime
    if isinstance(left, dtypes.Date) and isinstance(right, dtypes.Datetime):
        return right
    if isinstance(right, dtypes.Date) and isinstance(left, dtypes.Datetime):
        return left

    # Categorical/Enum + String -> String
    if (
        isinstance(left, dtypes.String)
        and (isinstance(right, (dtypes.Categorical, dtypes.Enum)))
    ) or (
        isinstance(right, dtypes.String)
        and (isinstance(left, (dtypes.Categorical, dtypes.Enum)))
    ):
        return dtypes.String()

    if isinstance(left, dtypes.Unknown) or isinstance(right, dtypes.Unknown):
        return dtypes.Unknown()

    return None
