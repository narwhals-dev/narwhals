# TODO @dangotbanned: Update this docstring
"""Deviations from `polars`.

- A `Selector` corresponds to a `nw.selectors` function
- Binary ops are represented as a `BinarySelector`, similar to `BinaryExpr`.
"""

from __future__ import annotations

import builtins
import re
from contextlib import suppress
from typing import TYPE_CHECKING, Any, ClassVar

from narwhals._plan._immutable import Immutable
from narwhals._plan.common import flatten_hash_safe
from narwhals._plan.exceptions import column_index_error, column_not_found_error
from narwhals._utils import Version, _parse_time_unit_and_time_zone
from narwhals.dtypes import DType, NumericType
from narwhals.typing import TimeUnit

if TYPE_CHECKING:
    from collections.abc import Container, Iterator
    from datetime import timezone
    from typing import TypeVar

    from narwhals._plan.expressions import SelectorIR
    from narwhals._plan.expressions.expr import RootSelector
    from narwhals._plan.schema import FrozenSchema
    from narwhals._plan.typing import OneOrIterable, Seq

    T = TypeVar("T")

_dtypes = Version.MAIN.dtypes

_ALL_TIME_UNITS = frozenset[TimeUnit](("ms", "us", "ns", "s"))


class Selector(Immutable):
    def to_selector_ir(self) -> RootSelector:
        from narwhals._plan.expressions.expr import RootSelector

        return RootSelector(selector=self)

    def to_dtype_selector(self) -> DTypeSelector:
        msg = f"expected datatype based expression got {self!r}"
        raise TypeError(msg)

    def matches_column(self, name: str, dtype: DType) -> bool:
        raise NotImplementedError(type(self))

    def into_columns(
        self, schema: FrozenSchema, ignored_columns: Container[str]
    ) -> Iterator[str]:
        # /// Turns the selector into an ordered set of selected columns from the schema.
        #
        # - The order of the columns corresponds to the order in the schema.
        # - Column names in `ignored_columns` are only used if they are explicitly mentioned by a `ByName` or `ByIndex`.
        # - `ignored_columns` are only evaluated against `All` and `Matches`
        # https://github.com/pola-rs/polars/blob/2b241543851800595efd343be016b65cdbdd3c9f/crates/polars-plan/src/dsl/selector.rs#L192-L193
        msg = f"{self.into_columns.__qualname__!r}"
        raise NotImplementedError(msg)


class DTypeSelector(Selector):
    # Will be updating things to be a bit closer to upstream
    # https://github.com/pola-rs/polars/blob/2b241543851800595efd343be016b65cdbdd3c9f/crates/polars-plan/src/dsl/selector.rs#L110-L172
    _dtype: ClassVar[type[DType]]

    def __init_subclass__(cls, *args: Any, dtype: type[DType], **kwds: Any) -> None:
        super().__init_subclass__(*args, **kwds)
        cls._dtype = dtype

    def __repr__(self) -> str:
        return f"ncs.{type(self).__name__.lower()}"

    def to_dtype_selector(self) -> DTypeSelector:
        return self

    def matches_column(self, name: str, dtype: DType) -> bool:
        return isinstance(dtype, self._dtype)

    def matches(self, dtype: DType) -> bool:
        # Exclusive to `DataTypeSelector`
        return isinstance(dtype, self._dtype)


class DTypeAll(DTypeSelector, dtype=DType):
    def __repr__(self) -> str:
        return "ncs.all()"

    def matches_column(self, name: str, dtype: DType) -> bool:
        return True

    def matches(self, dtype: DType) -> bool:
        return True


class All(Selector):
    # Both `Selector::Wildcard` and `DataTypeSelector::Wildcard` exist
    # Also `Empty`, but that's new
    def __repr__(self) -> str:
        return "ncs.all()"

    def to_dtype_selector(self) -> DTypeSelector:
        return DTypeAll()

    def matches_column(self, name: str, dtype: DType) -> bool:
        return True

    def into_columns(
        self, schema: FrozenSchema, ignored_columns: Container[str]
    ) -> Iterator[str]:
        if ignored_columns:
            yield from (name for name in schema if name not in ignored_columns)
        else:
            yield from schema


class ByIndex(Selector):
    # returns inputs in given order not in schema order.
    __slots__ = ("indices", "require_all")
    indices: Seq[int]
    require_all: bool

    @staticmethod
    def _iter_validate(indices: tuple[OneOrIterable[int], ...], /) -> Iterator[int]:
        for idx in flatten_hash_safe(indices):
            if not isinstance(idx, int):
                msg = f"invalid index value: {idx!r}"
                raise TypeError(msg)
            yield idx

    @staticmethod
    def from_indices(*indices: OneOrIterable[int], require_all: bool = True) -> ByIndex:
        return ByIndex(
            indices=tuple(ByIndex._iter_validate(indices)), require_all=require_all
        )

    @staticmethod
    def from_index(index: int, /, *, require_all: bool = True) -> ByIndex:
        return ByIndex(indices=(index,), require_all=require_all)

    def into_columns(
        self, schema: FrozenSchema, ignored_columns: Container[str]
    ) -> Iterator[str]:
        names = schema.names
        if not self.require_all:
            with suppress(IndexError):
                for index in self.indices:
                    yield names[index]
        else:
            n_fields = len(names)
            for index in self.indices:
                idx = index + n_fields if index < 0 else index
                if idx < 0 or idx >= n_fields:
                    raise column_index_error(index, schema)
                yield names[index]


class ByName(Selector):
    # returns inputs in given order not in schema order.
    __slots__ = ("names", "require_all")
    names: Seq[str]
    require_all: bool

    def __repr__(self) -> str:
        els = ", ".join(f"{nm!r}" for nm in self.names)
        return f"ncs.by_name({els})"

    @staticmethod
    def _iter_validate(names: tuple[OneOrIterable[str], ...], /) -> Iterator[str]:
        for name in flatten_hash_safe(names):
            if not isinstance(name, str):
                msg = f"invalid name: {name!r}"
                raise TypeError(msg)
            yield name

    @staticmethod
    def from_names(*names: OneOrIterable[str], require_all: bool = True) -> ByName:
        return ByName(names=tuple(ByName._iter_validate(names)), require_all=require_all)

    @staticmethod
    def from_name(name: str, /, *, require_all: bool = True) -> ByName:
        return ByName(names=(name,), require_all=require_all)

    def matches_column(self, name: str, dtype: DType) -> bool:
        return name in self.names

    def into_columns(
        self, schema: FrozenSchema, ignored_columns: Container[str]
    ) -> Iterator[str]:
        if not self.require_all:
            keys = schema.keys()
            yield from (name for name in self.names if name in keys)
        else:
            if not set(schema).issuperset(self.names):
                raise column_not_found_error(self.names, schema)
            yield from self.names


class Matches(Selector):
    __slots__ = ("pattern",)
    pattern: re.Pattern[str]

    @staticmethod
    def from_string(pattern: str, /) -> Matches:
        return Matches(pattern=re.compile(pattern))

    def __repr__(self) -> str:
        return f"ncs.matches(pattern={self.pattern.pattern!r})"

    def matches_column(self, name: str, dtype: DType) -> bool:
        return bool(self.pattern.search(name))

    def into_columns(
        self, schema: FrozenSchema, ignored_columns: Container[str]
    ) -> Iterator[str]:
        search = self.pattern.search
        if ignored_columns:
            for name in schema:
                if name not in ignored_columns and search(name):
                    yield name
        else:
            yield from (name for name in schema if search(name))


class Array(DTypeSelector, dtype=_dtypes.Array):
    __slots__ = ("inner", "size")
    inner: SelectorIR | None
    size: int | None
    """Not sure why polars is using the (`0.20.31`) deprecated name `width`."""

    def __repr__(self) -> str:
        inner = "" if not self.inner else repr(self.inner)
        size = self.size or "*"
        return f"ncs.array({inner}, size={size})"

    def matches_column(self, name: str, dtype: DType) -> bool:
        return (
            isinstance(dtype, _dtypes.Array)
            and (not (self.inner) or self.inner.matches_column(name, dtype.inner))  # type: ignore[arg-type]
            and (self.size is None or dtype.size == self.size)
        )


class Boolean(DTypeSelector, dtype=_dtypes.Boolean): ...


class ByDType(DTypeSelector, dtype=DType):
    __slots__ = ("dtypes",)
    dtypes: frozenset[DType | type[DType]]

    def __repr__(self) -> str:
        els = ", ".join(
            tp.__name__ if isinstance(tp, type) else repr(tp) for tp in self.dtypes
        )
        return f"ncs.by_dtype(dtypes=[{els}])"

    def matches_column(self, name: str, dtype: DType) -> bool:
        return dtype in self.dtypes

    @staticmethod
    def empty() -> ByDType:
        return ByDType(dtypes=frozenset())


class Categorical(DTypeSelector, dtype=_dtypes.Categorical): ...


class Datetime(DTypeSelector, dtype=_dtypes.Datetime):
    """Should swallow the [`utils` functions].

    Just re-wrapping them for now, since `CompliantSelectorNamespace` is still using them.

    [`utils` functions]: https://github.com/narwhals-dev/narwhals/blob/6d524ba04fca6fe2d6d25bdd69f75fabf1d79039/narwhals/utils.py#L1565-L1596
    """

    __slots__ = ("time_units", "time_zones")
    time_units: frozenset[TimeUnit]
    time_zones: frozenset[str | None]

    @staticmethod
    def from_time_unit_and_time_zone(
        time_unit: OneOrIterable[TimeUnit] | None,
        time_zone: OneOrIterable[str | timezone | None],
        /,
    ) -> Datetime:
        units, zones = _parse_time_unit_and_time_zone(time_unit, time_zone)
        return Datetime(time_units=frozenset(units), time_zones=frozenset(zones))

    def __repr__(self) -> str:
        return f"ncs.datetime(time_unit={builtins.list(self.time_units)}, time_zone={builtins.list(self.time_zones)})"

    def matches_column(self, name: str, dtype: DType) -> bool:
        units, zones = self.time_units, self.time_zones
        return (
            isinstance(dtype, _dtypes.Datetime)
            and (dtype.time_unit in units)
            and (
                dtype.time_zone in zones or ("*" in zones and dtype.time_zone is not None)
            )
        )


class Duration(DTypeSelector, dtype=_dtypes.Duration):
    __slots__ = ("time_units",)
    time_units: frozenset[TimeUnit]

    @staticmethod
    def from_time_unit(time_unit: OneOrIterable[TimeUnit] | None, /) -> Duration:
        if time_unit is None:
            units = _ALL_TIME_UNITS
        elif not isinstance(time_unit, str):
            units = frozenset(time_unit)
        else:
            units = frozenset((time_unit,))
        return Duration(time_units=units)

    def __repr__(self) -> str:
        return f"ncs.duration(time_unit={builtins.list(self.time_units)})"

    def matches_column(self, name: str, dtype: DType) -> bool:
        return isinstance(dtype, _dtypes.Duration) and (
            dtype.time_unit in self.time_units
        )


class Enum(DTypeSelector, dtype=_dtypes.Enum): ...


class List(DTypeSelector, dtype=_dtypes.List):
    __slots__ = ("inner",)
    inner: SelectorIR | None

    def __repr__(self) -> str:
        inner = "" if not self.inner else repr(self.inner)
        return f"ncs.list({inner})"

    def matches_column(self, name: str, dtype: DType) -> bool:
        return isinstance(dtype, _dtypes.List) and (
            not (self.inner) or self.inner.matches_column(name, dtype.inner)  # type: ignore[arg-type]
        )


class Numeric(DTypeSelector, dtype=NumericType): ...


class String(DTypeSelector, dtype=_dtypes.String): ...


class Struct(DTypeSelector, dtype=_dtypes.Struct): ...
