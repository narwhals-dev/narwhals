"""Deviations from `polars`.

- A `Selector` corresponds to a `nw.selectors` function
- Binary ops are represented as a `BinarySelector`, similar to `BinaryExpr`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from narwhals._plan._immutable import Immutable
from narwhals._plan.common import flatten_hash_safe
from narwhals._utils import Version, _parse_time_unit_and_time_zone

if TYPE_CHECKING:
    from datetime import timezone
    from typing import TypeVar

    from narwhals._plan import expr
    from narwhals._plan.expressions.expr import RootSelector
    from narwhals._plan.typing import OneOrIterable
    from narwhals.dtypes import DType
    from narwhals.typing import TimeUnit

    T = TypeVar("T")

dtypes = Version.MAIN.dtypes


class Selector(Immutable):
    def to_selector(self) -> RootSelector:
        from narwhals._plan.expressions.expr import RootSelector

        return RootSelector(selector=self)

    def matches_column(self, name: str, dtype: DType) -> bool:
        raise NotImplementedError(type(self))


class All(Selector):
    def __repr__(self) -> str:
        return "ncs.all()"

    def matches_column(self, name: str, dtype: DType) -> bool:
        return True


class ByDType(Selector):
    __slots__ = ("dtypes",)
    dtypes: frozenset[DType | type[DType]]

    @staticmethod
    def from_dtypes(*dtypes: OneOrIterable[DType | type[DType]]) -> ByDType:
        return ByDType(dtypes=frozenset(flatten_hash_safe(dtypes)))

    def __repr__(self) -> str:
        els = ", ".join(
            tp.__name__ if isinstance(tp, type) else repr(tp) for tp in self.dtypes
        )
        return f"ncs.by_dtype(dtypes=[{els}])"

    def matches_column(self, name: str, dtype: DType) -> bool:
        return dtype in self.dtypes


class Boolean(Selector):
    def __repr__(self) -> str:
        return "ncs.boolean()"

    def matches_column(self, name: str, dtype: DType) -> bool:
        return isinstance(dtype, dtypes.Boolean)


class Categorical(Selector):
    def __repr__(self) -> str:
        return "ncs.categorical()"

    def matches_column(self, name: str, dtype: DType) -> bool:
        return isinstance(dtype, dtypes.Categorical)


class Datetime(Selector):
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
        return f"ncs.datetime(time_unit={list(self.time_units)}, time_zone={list(self.time_zones)})"

    def matches_column(self, name: str, dtype: DType) -> bool:
        units, zones = self.time_units, self.time_zones
        return (
            isinstance(dtype, dtypes.Datetime)
            and (dtype.time_unit in units)
            and (
                dtype.time_zone in zones or ("*" in zones and dtype.time_zone is not None)
            )
        )


class Matches(Selector):
    __slots__ = ("pattern",)
    pattern: re.Pattern[str]

    @staticmethod
    def from_string(pattern: str, /) -> Matches:
        return Matches(pattern=re.compile(pattern))

    @staticmethod
    def from_names(*names: OneOrIterable[str]) -> Matches:
        """Implements `cs.by_name` to support `__r<op>__` with column selections."""
        it = flatten_hash_safe(names)
        return Matches.from_string(f"^({'|'.join(re.escape(name) for name in it)})$")

    def __repr__(self) -> str:
        return f"ncs.matches(pattern={self.pattern.pattern!r})"

    def matches_column(self, name: str, dtype: DType) -> bool:
        return bool(self.pattern.search(name))


class Numeric(Selector):
    def __repr__(self) -> str:
        return "ncs.numeric()"

    def matches_column(self, name: str, dtype: DType) -> bool:
        return dtype.is_numeric()


class String(Selector):
    def __repr__(self) -> str:
        return "ncs.string()"

    def matches_column(self, name: str, dtype: DType) -> bool:
        return isinstance(dtype, dtypes.String)


def all() -> expr.Selector:
    return All().to_selector().to_narwhals()


def by_dtype(*dtypes: OneOrIterable[DType | type[DType]]) -> expr.Selector:
    return ByDType.from_dtypes(*dtypes).to_selector().to_narwhals()


def by_name(*names: OneOrIterable[str]) -> expr.Selector:
    return Matches.from_names(*names).to_selector().to_narwhals()


def boolean() -> expr.Selector:
    return Boolean().to_selector().to_narwhals()


def categorical() -> expr.Selector:
    return Categorical().to_selector().to_narwhals()


def datetime(
    time_unit: OneOrIterable[TimeUnit] | None = None,
    time_zone: OneOrIterable[str | timezone | None] = ("*", None),
) -> expr.Selector:
    return (
        Datetime.from_time_unit_and_time_zone(time_unit, time_zone)
        .to_selector()
        .to_narwhals()
    )


def matches(pattern: str) -> expr.Selector:
    return Matches.from_string(pattern).to_selector().to_narwhals()


def numeric() -> expr.Selector:
    return Numeric().to_selector().to_narwhals()


def string() -> expr.Selector:
    return String().to_selector().to_narwhals()
