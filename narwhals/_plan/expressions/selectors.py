"""Expressions for selecting columns."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Generic, final

import narwhals.dtypes as nw_dtypes
from narwhals._plan._expr_ir import ExprIR, SelectorIR
from narwhals._plan.exceptions import (
    column_index_error,
    column_not_found_error,
    one_or_iterable_type_error,
)
from narwhals._plan.typing import (
    LeftSelectorT_co,
    RightSelectorT_co,
    SelectorOperatorT,
    SelectorT_co,
)
from narwhals._utils import _parse_time_unit_and_time_zone
from narwhals.dtypes import DType, FloatType, IntegerType, NumericType, TemporalType
from narwhals.typing import IntoDType, TimeUnit

if TYPE_CHECKING:
    from collections.abc import Iterator
    from datetime import timezone

    from typing_extensions import Self

    from narwhals._plan.typing import Ignored, OneOrIterable, Seq


_ALL_TIME_UNITS = frozenset[TimeUnit](("ms", "us", "ns", "s"))


class RootSelector(SelectorIR):
    """A single selector expression."""

    def __repr__(self) -> str:
        return f"ncs.{type(self).__name__.lower()}()"


@final
class BinarySelector(
    SelectorIR, Generic[LeftSelectorT_co, SelectorOperatorT, RightSelectorT_co]
):
    """A set operation applied to two selectors.

    Important:
        Matches are returned in schema order.
    """

    __slots__ = ("left", "op", "right")
    left: LeftSelectorT_co  # type: ignore[misc]
    op: SelectorOperatorT
    right: RightSelectorT_co  # type: ignore[misc]

    def __repr__(self) -> str:
        return f"[{self.left!r} {self.op!r} {self.right!r}]"

    def iter_expand_selector(
        self, schema: Mapping[str, DType], ignored_columns: Ignored = (), /
    ) -> Iterator[str]:
        left = frozenset(self.left.iter_expand_selector(schema, ignored_columns))
        right = frozenset(self.right.iter_expand_selector(schema, ignored_columns))
        keep: frozenset[str]
        if (keep := self.op(left, right)) and len(keep) == len(schema):
            yield from schema
        elif keep:
            yield from (nm for nm in schema if nm in keep)

    def _matches_dtype(self, dtype: IntoDType) -> bool:
        left = self.left.matches(dtype)
        right = self.right.matches(dtype)
        return bool(self.op(left, right))

    def to_dtype_selector(
        self,
    ) -> Self | BinarySelector[SelectorIR, SelectorOperatorT, SelectorIR]:
        left, right = self.left.to_dtype_selector(), self.right.to_dtype_selector()
        if left is self.left and right is self.right:
            return self
        return BinarySelector(left=left, op=self.op, right=right)


@final
class InvertSelector(SelectorIR, Generic[SelectorT_co]):
    """The complement of a selector.

    Important:
        Matches are returned in schema order.

    Arguments:
        selector: A selector to invert.
    """

    __slots__ = ("selector",)
    selector: SelectorT_co  # type: ignore[misc]

    def __repr__(self) -> str:
        return f"~{self.selector!r}"

    def iter_expand_selector(
        self, schema: Mapping[str, DType], ignored_columns: Ignored = (), /
    ) -> Iterator[str]:
        expand = self.selector.iter_expand_selector
        if not (ignore := frozenset(expand(schema, ignored_columns))):
            yield from schema
        elif len(ignore) != len(schema):
            yield from (nm for nm in schema if nm not in ignore)

    def _matches_dtype(self, dtype: IntoDType) -> bool:
        return not self.selector.matches(dtype)

    def to_dtype_selector(self) -> Self | InvertSelector[SelectorIR]:
        s = self.selector
        if (after := s.to_dtype_selector()) is s:
            return self
        return InvertSelector(selector=after)

    def invert(self) -> SelectorT_co:
        return self.selector


class DTypeSelector(RootSelector):
    """A selector that (exclusively) operates on data types.

    Important:
        Matches are returned in schema order.

    Adapted from [upstream].

    [upstream]: https://github.com/pola-rs/polars/blob/2b241543851800595efd343be016b65cdbdd3c9f/crates/polars-plan/src/dsl/selector.rs#L110-L172
    """

    _dtype: ClassVar[type[DType]]

    def __init_subclass__(cls: type[Self], *, selects: type[DType], **_: Any) -> None:
        super().__init_subclass__(**_)
        cls._dtype = selects

    def to_dtype_selector(self) -> Self:
        return self

    def _matches_dtype(self, dtype: IntoDType) -> bool:
        # NOTE @dangotbanned: Coverage is flaky (somehow?) for the `issubclass` branch
        return (
            issubclass(dtype, self._dtype)
            if isinstance(dtype, type)
            else isinstance(dtype, self._dtype)
        )

    def iter_expand_selector(
        self, schema: Mapping[str, DType], ignored_columns: Ignored = (), /
    ) -> Iterator[str]:
        if ignored_columns:
            for name, dtype in schema.items():
                if self.matches(dtype) and name not in ignored_columns:
                    yield name
        else:
            yield from (name for name, dtype in schema.items() if self.matches(dtype))


class AllDType(DTypeSelector, selects=DType):
    def __repr__(self) -> str:
        return "ncs.all()"

    def _matches_dtype(self, dtype: IntoDType) -> bool:
        return True


class All(RootSelector):
    """Select all columns.

    Important:
        Matches are returned in schema order.

    ## Examples
    Represents both variants of `all()`:
    >>> import narwhals._plan as nw
    >>> import narwhals._plan.selectors as ncs
    >>> expr = nw.all()
    >>> expr._ir
    ncs.all()
    >>> selector = ncs.all()
    >>> selector._ir
    ncs.all()

    While they are *internally* identical:
    >>> expr._ir == selector._ir
    True

    The *outer* container defines binary operator behavior:
    >>> not isinstance(expr, nw.Selector)
    True
    >>> isinstance(selector, nw.Selector)
    True
    """

    def to_dtype_selector(self) -> AllDType:
        return AllDType()

    def iter_expand_selector(
        self, schema: Mapping[str, DType], ignored_columns: Ignored = (), /
    ) -> Iterator[str]:
        if ignored_columns:
            yield from (name for name in schema if name not in ignored_columns)
        else:
            yield from schema

    def invert(self) -> Empty:
        return Empty()


# TODO @dangotbanned: Singletons for `All`, `Empty`, + dtype equivalent
class Empty(RootSelector):
    """Select no columns.

    Important:
        Matches nothing, including when used in `group_by`.
    """

    def to_dtype_selector(self) -> EmptyDType:
        return EmptyDType()

    def iter_expand_selector(
        self, _: Mapping[str, DType], __: Ignored = (), /
    ) -> Iterator[str]:
        # NOTE: https://github.com/pola-rs/polars/blob/7fc9f1875714fe9893c4d849b9593c1e4db1e854/crates/polars-plan/src/dsl/selector.rs#L274
        yield from ()

    def invert(self) -> All:
        return All()


class EmptyDType(DTypeSelector, selects=DType):
    def __repr__(self) -> str:
        return "ncs.empty()"

    def _matches_dtype(self, dtype: IntoDType) -> bool:
        return False


class ByIndex(RootSelector):
    """Select all columns matching the given indices.

    Important:
        Matches are returned in the order declared in `indices`.

    Arguments:
        indices: Positions of columns to select.
            Negative indexing is supported.
        require_all: Raise (default) if any index is out-of-bounds.
            If set to `False`, out-of-bounds indices are ignored.

    Examples:
        We've got 4 ways to get here:
        >>> import narwhals._plan as nw
        >>> import narwhals._plan.selectors as ncs

        Only one starts it's life as an expression:
        >>> expr = nw.nth(5)
        >>> expr._ir
        ncs.by_index([5], require_all=True)
        >>> not isinstance(expr, nw.Selector)
        True

        The rest are selectors all the way down:
        >>> selector = ncs.by_index(0, 1)
        >>> selector._ir
        ncs.by_index([0, 1], require_all=True)
        >>> isinstance(selector, nw.Selector)
        True

        With these two being pure syntax sugar:
        >>> first = ncs.first()
        >>> last = ncs.last()
        >>> first._ir
        ncs.first()
        >>> print(first._ir)
        ByIndex(indices=[0], require_all=True)
        >>> print(last._ir)
        ByIndex(indices=[-1], require_all=True)
    """

    # TODO @dangotbanned: (low-priority) Specialize a `ByIndex` that preserves `range` instead of converting to `tuple`
    # - hashable
    # - runtime guarantees `Sequence[int]` (less work for us)
    # - expansion loop doesn't need to `abs` every element
    #   - given we know ahead of time it is mono asc/desc and the start/end
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
        for outer in indices:
            if isinstance(outer, int):
                yield outer
            elif not isinstance(outer, Iterable):
                raise one_or_iterable_type_error("index", outer)
            elif isinstance(outer, range):
                yield from outer
            else:
                for inner in outer:
                    if isinstance(inner, int):
                        yield inner
                    else:
                        raise one_or_iterable_type_error(inner, outer)

    @staticmethod
    def from_indices(*indices: OneOrIterable[int], require_all: bool = True) -> ByIndex:
        return ByIndex(
            indices=tuple(ByIndex._iter_validate(indices)), require_all=require_all
        )

    @staticmethod
    def from_index(index: int, /, *, require_all: bool = True) -> ByIndex:
        return ByIndex(indices=(index,), require_all=require_all)

    def iter_expand_selector(
        self, schema: Mapping[str, DType], _: Ignored = (), /
    ) -> Iterator[str]:
        names = tuple(schema)
        n_fields = len(names)
        if not self.require_all:
            if n_fields != 0:
                yield from (names[idx] for idx in self.indices if abs(idx) < n_fields)
        else:
            for idx in self.indices:
                if abs(idx) < n_fields:
                    yield names[idx]
                else:
                    raise column_index_error(idx, schema)


class ByName(RootSelector):
    """Select all columns matching the given names.

    Important:
        Matches are returned in the order declared in `names`.

    Arguments:
        names: Names of columns to select.
        require_all: Raise (default) if any name did not match.
            If set to `False`, failed matches are ignored.

    Examples:
        >>> import narwhals._plan as nw
        >>> import narwhals._plan.selectors as ncs
        >>> selector = ncs.by_name("one", "two")
        >>> selector._ir
        ncs.by_name('one', 'two', require_all=True)

        Also represents multi-output `col(...)`:
        >>> expr = nw.col("one", "two")
        >>> print(expr._ir)
        ByName(names=['one', 'two'], require_all=True)
    """

    __slots__ = ("names", "require_all")
    names: Seq[str]
    require_all: bool

    def __repr__(self) -> str:
        els = ", ".join(f"{nm!r}" for nm in self.names)
        return f"ncs.by_name({els}, require_all={self.require_all})"

    @staticmethod
    def _iter_validate(names: tuple[OneOrIterable[str], ...], /) -> Iterator[str]:
        for outer in names:
            if isinstance(outer, str):
                yield outer
            elif isinstance(outer, Iterable):
                for inner in outer:
                    if isinstance(inner, str):
                        yield inner
                    else:
                        raise one_or_iterable_type_error("name", inner, outer)
            else:
                raise one_or_iterable_type_error("name", outer)

    @staticmethod
    def from_names(*names: OneOrIterable[str], require_all: bool = True) -> ByName:
        return ByName(names=tuple(ByName._iter_validate(names)), require_all=require_all)

    @staticmethod
    def from_name(name: str, /, *, require_all: bool = True) -> ByName:
        return ByName(names=(name,), require_all=require_all)

    def iter_expand_selector(
        self, schema: Mapping[str, DType], _: Ignored = (), /
    ) -> Iterator[str]:
        if not self.require_all:
            keys = schema.keys()
            yield from (name for name in self.names if name in keys)
        else:
            if not set(schema).issuperset(self.names):
                raise column_not_found_error(self.names, schema)
            yield from self.names

    def iter_output_name(self) -> Iterator[ExprIR]:
        # NOTE: Valid for `ByName.from_name`-only
        yield self


class Matches(RootSelector):
    """Select columns that match the given regex pattern.

    Important:
        Matches are returned in schema order.

    Arguments:
        pattern: A compiled regular expression object.
            Uses [`re.search`] which matches *anywhere* in the column name,
            if anchors are not provided.

    [`re.search`]: https://docs.python.org/3/library/re.html#re.search
    """

    __slots__ = ("pattern",)
    pattern: re.Pattern[str]

    @staticmethod
    def from_string(pattern: str, /) -> Matches:
        return Matches(pattern=re.compile(pattern))

    def __repr__(self) -> str:
        return f"ncs.matches({self.pattern.pattern!r})"

    def iter_expand_selector(
        self, schema: Mapping[str, DType], ignored_columns: Ignored = (), /
    ) -> Iterator[str]:
        search = self.pattern.search
        if ignored_columns:
            for name in schema:
                if name not in ignored_columns and search(name):
                    yield name
        else:
            yield from (name for name in schema if search(name))


class Array(DTypeSelector, selects=nw_dtypes.Array):
    __slots__ = ("inner", "size")
    inner: SelectorIR | None
    size: int | None

    def __repr__(self) -> str:
        inner = "" if not self.inner else repr(self.inner)
        size = self.size or "*"
        return f"ncs.array({inner}, size={size})"

    def _matches_dtype(self, dtype: IntoDType) -> bool:
        return (
            isinstance(dtype, nw_dtypes.Array)
            and ((s := self.inner) is None or s.matches(dtype.inner))
            and (self.size is None or dtype.size == self.size)
        )


class List(DTypeSelector, selects=nw_dtypes.List):
    __slots__ = ("inner",)
    inner: SelectorIR | None

    def __repr__(self) -> str:
        inner = "" if not self.inner else repr(self.inner)
        return f"ncs.list({inner})"

    def _matches_dtype(self, dtype: IntoDType) -> bool:
        return isinstance(dtype, nw_dtypes.List) and (
            (s := self.inner) is None or s.matches(dtype.inner)
        )


class ByDType(DTypeSelector, selects=DType):
    __slots__ = ("dtypes",)
    dtypes: frozenset[DType | type[DType]]

    def __repr__(self) -> str:
        return f"ncs.by_dtype([{', '.join(sorted(map(repr, self.dtypes)))}])"

    def _matches_dtype(self, dtype: DType | type[DType]) -> bool:
        return dtype in self.dtypes


class Datetime(DTypeSelector, selects=nw_dtypes.Datetime):
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

    def _matches_dtype(self, dtype: IntoDType) -> bool:
        units, zones = self.time_units, self.time_zones
        return (
            isinstance(dtype, (nw_dtypes.Datetime, type(nw_dtypes.Datetime)))
            and (dtype.time_unit in units)
            and (
                dtype.time_zone in zones or ("*" in zones and dtype.time_zone is not None)
            )
        )


class Duration(DTypeSelector, selects=nw_dtypes.Duration):
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

    def _matches_dtype(self, dtype: IntoDType) -> bool:
        return isinstance(dtype, (nw_dtypes.Duration, type(nw_dtypes.Duration))) and (
            dtype.time_unit in self.time_units
        )


# fmt: off
class Boolean(DTypeSelector, selects=nw_dtypes.Boolean): ...
class Categorical(DTypeSelector, selects=nw_dtypes.Categorical): ...
class Decimal(DTypeSelector, selects=nw_dtypes.Decimal): ...
class Enum(DTypeSelector, selects=nw_dtypes.Enum): ...
class Float(DTypeSelector, selects=FloatType): ...
class Integer(DTypeSelector, selects=IntegerType): ...
class Numeric(DTypeSelector, selects=NumericType): ...
class String(DTypeSelector, selects=nw_dtypes.String): ...
class Struct(DTypeSelector, selects=nw_dtypes.Struct): ...
class Temporal(DTypeSelector, selects=TemporalType): ...
# fmt: on
