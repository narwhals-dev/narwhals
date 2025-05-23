"""Deviations from `polars`.

- A `Selector` corresponds to a `nw.selectors` function
- Binary ops are represented as a subtype of `BinaryExpr`
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterable

from narwhals._plan.common import Immutable, is_iterable_reject
from narwhals.utils import _parse_time_unit_and_time_zone

if TYPE_CHECKING:
    from datetime import timezone
    from typing import Iterator, TypeVar

    from narwhals._plan.dummy import DummySelector
    from narwhals._plan.expr import RootSelector
    from narwhals.dtypes import DType
    from narwhals.typing import TimeUnit

    T = TypeVar("T")


class Selector(Immutable):
    def to_selector(self) -> RootSelector:
        from narwhals._plan.expr import RootSelector

        return RootSelector(selector=self)


class All(Selector): ...


class ByDType(Selector):
    __slots__ = ("dtypes",)

    dtypes: frozenset[DType | type[DType]]

    @staticmethod
    def from_dtypes(
        *dtypes: DType | type[DType] | Iterable[DType | type[DType]],
    ) -> ByDType:
        return ByDType(dtypes=frozenset(_flatten_hash_safe(dtypes)))


class Boolean(Selector): ...


class Categorical(Selector): ...


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
        time_unit: TimeUnit | Iterable[TimeUnit] | None,
        time_zone: str | timezone | Iterable[str | timezone | None] | None,
        /,
    ) -> Datetime:
        units, zones = _parse_time_unit_and_time_zone(time_unit, time_zone)
        return Datetime(time_units=frozenset(units), time_zones=frozenset(zones))


class Matches(Selector):
    __slots__ = ("pattern",)

    pattern: re.Pattern[str]

    @staticmethod
    def from_string(pattern: str, /) -> Matches:
        return Matches(pattern=re.compile(pattern))


class Numeric(Selector): ...


class String(Selector): ...


def all() -> DummySelector:
    return All().to_selector().to_narwhals()


def by_dtype(
    *dtypes: DType | type[DType] | Iterable[DType | type[DType]],
) -> DummySelector:
    return ByDType.from_dtypes(*dtypes).to_selector().to_narwhals()


def boolean() -> DummySelector:
    return Boolean().to_selector().to_narwhals()


def categorical() -> DummySelector:
    return Categorical().to_selector().to_narwhals()


def datetime(
    time_unit: TimeUnit | Iterable[TimeUnit] | None = None,
    time_zone: str | timezone | Iterable[str | timezone | None] | None = ("*", None),
) -> DummySelector:
    return (
        Datetime.from_time_unit_and_time_zone(time_unit, time_zone)
        .to_selector()
        .to_narwhals()
    )


def matches(pattern: str) -> DummySelector:
    return Matches.from_string(pattern).to_selector().to_narwhals()


def numeric() -> DummySelector:
    return Numeric().to_selector().to_narwhals()


def string() -> DummySelector:
    return String().to_selector().to_narwhals()


def _flatten_hash_safe(iterable: Iterable[T | Iterable[T]], /) -> Iterator[T]:
    """Fully unwrap all levels of nesting.

    Aiming to reduce the chances of passing an unhashable argument.
    """
    for element in iterable:
        if isinstance(element, Iterable) and not is_iterable_reject(element):
            yield from _flatten_hash_safe(element)
        else:
            yield element  # type: ignore[misc]
