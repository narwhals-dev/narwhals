from __future__ import annotations

import functools
import re
from typing import TYPE_CHECKING, Any, ClassVar, final

from narwhals._plan._immutable import Immutable
from narwhals._plan.common import flatten_hash_safe
from narwhals._plan.exceptions import column_index_error, column_not_found_error
from narwhals._utils import (
    Version,
    _parse_time_unit_and_time_zone,
    isinstance_or_issubclass,
)
from narwhals.dtypes import DType, FloatType, IntegerType, NumericType, TemporalType
from narwhals.typing import IntoDType, TimeUnit

if TYPE_CHECKING:
    from collections.abc import Iterator
    from datetime import timezone

    import narwhals.dtypes as nw_dtypes
    from narwhals._plan.expressions import SelectorIR
    from narwhals._plan.expressions.expr import RootSelector
    from narwhals._plan.schema import FrozenSchema
    from narwhals._plan.typing import Ignored, OneOrIterable, Seq


_dtypes = Version.MAIN.dtypes

_ALL_TIME_UNITS = frozenset[TimeUnit](("ms", "us", "ns", "s"))


class Selector(Immutable):
    def __repr__(self) -> str:
        return f"ncs.{type(self).__name__.lower()}()"

    def to_selector_ir(self) -> RootSelector:
        from narwhals._plan.expressions.expr import RootSelector

        return RootSelector(selector=self)

    def to_dtype_selector(self) -> DTypeSelector:
        msg = f"expected datatype based expression got {self!r}"
        raise TypeError(msg)

    def iter_expand(
        self, schema: FrozenSchema, ignored_columns: Ignored
    ) -> Iterator[str]:
        """Yield column names that match the selector, in `schema` order[^1].

        Adapted from [upstream].

        Arguments:
            schema: Target scope to expand the selector in.
            ignored_columns: Names of `group_by` columns, which are excluded[^2] from the result.

        Note:
            [^1]: `ByName`, `ByIndex` return their inputs in given order not in schema order.

        Note:
            [^2]: `ByName`, `ByIndex` will never be ignored.

        [upstream]: https://github.com/pola-rs/polars/blob/2b241543851800595efd343be016b65cdbdd3c9f/crates/polars-plan/src/dsl/selector.rs#L188-L198
        """
        msg = f"{type(self).__name__}.iter_expand"
        raise NotImplementedError(msg)


class DTypeSelector(Selector):
    # https://github.com/pola-rs/polars/blob/2b241543851800595efd343be016b65cdbdd3c9f/crates/polars-plan/src/dsl/selector.rs#L110-L172
    _dtype: ClassVar[type[DType]]

    def __init_subclass__(cls, *args: Any, dtype: type[DType], **kwds: Any) -> None:
        super().__init_subclass__(*args, **kwds)
        cls._dtype = dtype

    def to_dtype_selector(self) -> DTypeSelector:
        return self

    @final
    def matches(self, dtype: IntoDType) -> bool:
        """Return True if we can select this dtype.

        Important:
            The result will *only* be cached if this method is **not overridden**.
            Instead, use `DTypeSelector._matches` to customize the check.
        """
        return _selector_matches(self, dtype)

    def _matches(self, dtype: IntoDType) -> bool:
        """Implementation of `DTypeSelector.matches`."""
        return isinstance_or_issubclass(dtype, self._dtype)

    def iter_expand(
        self, schema: FrozenSchema, ignored_columns: Ignored
    ) -> Iterator[str]:
        if ignored_columns:
            for name, dtype in schema.items():
                if self.matches(dtype) and name not in ignored_columns:
                    yield name
        else:
            yield from (name for name, dtype in schema.items() if self.matches(dtype))


class DTypeAll(DTypeSelector, dtype=DType):
    def __repr__(self) -> str:
        return "ncs.all()"

    def _matches(self, dtype: IntoDType) -> bool:
        return True


class All(Selector):
    def to_dtype_selector(self) -> DTypeSelector:
        return DTypeAll()

    def iter_expand(
        self, schema: FrozenSchema, ignored_columns: Ignored
    ) -> Iterator[str]:
        if ignored_columns:
            yield from (name for name in schema if name not in ignored_columns)
        else:
            yield from schema


class ByIndex(Selector):
    __slots__ = ("indices", "require_all")
    indices: Seq[int]
    require_all: bool

    def __repr__(self) -> str:
        if len(self.indices) == 1 and self.indices[0] in {0, -1}:
            name = "first" if self.indices[0] == 0 else "last"
            return f"ncs.{name}()"
        return f"ncs.by_index({list(self.indices)}, require_all={self.require_all})"

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

    def iter_expand(
        self, schema: FrozenSchema, ignored_columns: Ignored
    ) -> Iterator[str]:
        names = schema.names
        n_fields = len(names)
        if not self.require_all:
            if n_fields == 0:
                yield from ()
            else:
                yield from (names[idx] for idx in self.indices if abs(idx) < n_fields)
        else:
            for idx in self.indices:
                if abs(idx) < n_fields:
                    yield names[idx]
                else:
                    raise column_index_error(idx, schema)


class ByName(Selector):
    __slots__ = ("names", "require_all")
    names: Seq[str]
    require_all: bool

    def __repr__(self) -> str:
        els = ", ".join(f"{nm!r}" for nm in self.names)
        return f"ncs.by_name({els}, require_all={self.require_all})"

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

    def iter_expand(
        self, schema: FrozenSchema, ignored_columns: Ignored
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
        return f"ncs.matches({self.pattern.pattern!r})"

    def iter_expand(
        self, schema: FrozenSchema, ignored_columns: Ignored
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

    def __repr__(self) -> str:
        inner = "" if not self.inner else repr(self.inner)
        size = self.size or "*"
        return f"ncs.array({inner}, size={size})"

    def _matches(self, dtype: IntoDType) -> bool:
        return (
            isinstance(dtype, _dtypes.Array)
            and _inner_selector_matches(self, dtype)
            and (self.size is None or dtype.size == self.size)
        )


class Boolean(DTypeSelector, dtype=_dtypes.Boolean): ...


class ByDType(DTypeSelector, dtype=DType):
    __slots__ = ("dtypes",)
    dtypes: frozenset[DType | type[DType]]

    def __repr__(self) -> str:
        if not self.dtypes:
            return "ncs.empty()"
        return f"ncs.by_dtype([{', '.join(sorted(map(repr, self.dtypes)))}])"

    def _matches(self, dtype: DType | type[DType]) -> bool:
        return dtype in self.dtypes

    @staticmethod
    def empty() -> ByDType:
        return ByDType(dtypes=frozenset())


class Categorical(DTypeSelector, dtype=_dtypes.Categorical): ...


class Datetime(DTypeSelector, dtype=_dtypes.Datetime):
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
        time_unit = "*" if self.time_units == _ALL_TIME_UNITS else list(self.time_units)
        time_zone = "*" if self.time_zones == {"*", None} else list(self.time_zones)
        return f"ncs.datetime(time_unit={time_unit}, time_zone={time_zone})"

    def _matches(self, dtype: IntoDType) -> bool:
        units, zones = self.time_units, self.time_zones
        return (
            isinstance_or_issubclass(dtype, _dtypes.Datetime)
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
        time_unit = "*" if self.time_units == _ALL_TIME_UNITS else list(self.time_units)
        return f"ncs.duration(time_unit={time_unit})"

    def _matches(self, dtype: IntoDType) -> bool:
        return isinstance_or_issubclass(dtype, _dtypes.Duration) and (
            dtype.time_unit in self.time_units
        )


class Enum(DTypeSelector, dtype=_dtypes.Enum): ...


class Decimal(DTypeSelector, dtype=_dtypes.Decimal): ...


class Float(DTypeSelector, dtype=FloatType): ...


class Integer(DTypeSelector, dtype=IntegerType): ...


class List(DTypeSelector, dtype=_dtypes.List):
    __slots__ = ("inner",)
    inner: SelectorIR | None

    def __repr__(self) -> str:
        inner = "" if not self.inner else repr(self.inner)
        return f"ncs.list({inner})"

    def _matches(self, dtype: IntoDType) -> bool:
        return isinstance(dtype, _dtypes.List) and _inner_selector_matches(self, dtype)


class Numeric(DTypeSelector, dtype=NumericType): ...


class String(DTypeSelector, dtype=_dtypes.String): ...


class Struct(DTypeSelector, dtype=_dtypes.Struct): ...


class Temporal(DTypeSelector, dtype=TemporalType): ...


@functools.lru_cache(maxsize=128)
def _selector_matches(selector: DTypeSelector, dtype: IntoDType, /) -> bool:
    # `DTypeSelector.matches` (uncached)
    #   -> `_selector_matches` (cached)
    #   -> `DTypeSelector._matches` (impl)
    return selector._matches(dtype)


def _inner_selector_matches(
    selector: Array | List, dtype: nw_dtypes.Array | nw_dtypes.List
) -> bool:
    return selector.inner is None or selector.inner.matches(dtype.inner)
