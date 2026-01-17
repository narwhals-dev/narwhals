"""Adapted from [@FBruzzesi script (2026-01-11)].

[@FBruzzesi script (2026-01-11)]: https://github.com/narwhals-dev/narwhals/pull/3396#issuecomment-3733465005
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from itertools import chain, product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Generic, overload

import polars as pl

from narwhals._typing_compat import TypeVar
from narwhals.dtypes import (
    Array,
    Binary,
    Boolean,
    Categorical,
    Date,
    Datetime,
    Decimal,
    DType,
    Duration,
    Enum,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    List,
    NumericType,
    Object,
    String,
    Struct,
    Time,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    Unknown,
)
from narwhals.dtypes._supertyping import get_supertype

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from datetime import timezone

    from typing_extensions import Self, TypeAlias

    from narwhals.typing import TimeUnit

Todo: TypeAlias = Any
"""Placeholder type for planning interactions."""


T = TypeVar("T")
DESTINATION_PATH: Final[Path] = Path("docs") / "concepts" / "promotion-rules.md"

Categories: TypeAlias = list[str]
TimeZone: TypeAlias = "str | timezone | None"

Values: TypeAlias = "Categories | TimeUnit | TimeZone"


def get_leaf_subclasses(cls: type[T]) -> list[type[T]]:
    """Get all leaf subclasses (classes with no further subclasses)."""
    leaves = []
    for subclass in cls.__subclasses__():
        if subclass.__subclasses__():  # Has children, recurse
            leaves.extend(get_leaf_subclasses(subclass))
        else:  # No children, it's a "leaf"
            leaves.append(subclass)
    return leaves


def collect_supertypes() -> None:
    from narwhals.dtypes import _classes as _classes, _classes_v1 as _classes_v1  # noqa: I001, PLC0414

    dtypes = get_leaf_subclasses(DType)
    supertypes: list[tuple[str, str, str]] = []
    for left, right in product(dtypes, dtypes):
        promoted: str
        base_types = frozenset((left, right))
        left_str, right_str = str(left), str(right)

        if Unknown in base_types:
            promoted = str(Unknown)
        elif left is right:
            promoted = str(left)
        elif left.is_nested() or right.is_nested():
            promoted = ""
        else:
            if left is Enum:
                left = Enum(["tmp"])  # noqa: PLW2901
            if right is Enum:
                right = Enum(["tmp"])  # noqa: PLW2901

            _promoted = get_supertype(left(), right())
            promoted = str(_promoted.__class__) if _promoted else ""

        supertypes.append((left_str, right_str, promoted))

    frame = (
        pl.DataFrame(supertypes, schema=["_", "right", "supertype"], orient="row")
        .pivot(
            index="_",
            on="right",
            values="supertype",
            aggregate_function=None,
            sort_columns=True,
        )
        .sort("_")
        .rename({"_": ""})
    )

    with (
        pl.Config(
            tbl_rows=30,
            tbl_cols=30,
            tbl_formatting="MARKDOWN",
            tbl_hide_column_data_types=True,
            tbl_hide_dataframe_shape=True,
            tbl_cell_alignment="LEFT",
            tbl_width_chars=-1,
        ),
        DESTINATION_PATH.open(mode="w", encoding="utf-8", newline="\n") as file,
    ):
        file.write(str(frame))
        file.write("\n")


@dataclass(frozen=True)
class _Param:
    name: str
    values: tuple[Any, ...]

    def __len__(self) -> int:
        return len(self.values)

    def to_params(self) -> _Params:
        return _Params(self)


class _Params:
    _params: tuple[_Param, ...]

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self._params!r}"

    def __len__(self) -> int:
        return len(self._params)

    def __init__(self, *params: _Param) -> None:
        self._params = params

    def parametrize(self, parameter_name: str, *values: Values) -> _Params:
        """Chain an additional parametrization to this one.

        Arguments:
            parameter_name: The (singular) name of the parameter target.
            *values: Individual values.
        """
        return _Params(*self._params, _Param(parameter_name, values))

    def _names(self) -> Iterator[str]:
        yield from (p.name for p in self._params)

    def _values(self) -> Iterator[tuple[Any, ...]]:
        yield from (p.values for p in self._params)

    def iter_bind_product(
        self, signature: inspect.Signature
    ) -> Iterator[inspect.BoundArguments]:
        if len(self) == 1:
            only = self._params[0]
            name = only.name
            for value in only.values:
                yield signature.bind(**{name: value})
        else:
            names = tuple(self._names())
            for parameter_set in product(*self._values()):
                yield signature.bind(**dict(zip(names, parameter_set)))


def parametrize(parameter_name: str, *values: Values) -> _Params:
    """Define the set of values for a given parameter.

    Arguments:
        parameter_name: The (singular) name of the parameter target.
        *values: Individual values.
    """
    return _Param(parameter_name, values).to_params()


_Parametric: TypeAlias = "Datetime | Duration | Enum"
_Nested: TypeAlias = "Array | List | Struct"
_Singleton: TypeAlias = "Binary | Boolean | Categorical | Date | Time | NumericType | String | Object | Unknown"

D = TypeVar("D", bound=DType, default=DType)
D_co = TypeVar("D_co", bound=DType, covariant=True)
DS_co = TypeVar("DS_co", bound=_Singleton, covariant=True)
DP_co = TypeVar("DP_co", bound=_Parametric, covariant=True)
DN_co = TypeVar("DN_co", bound=_Nested, covariant=True)
I_co = TypeVar("I_co", bound="DTypeProxy[DType]", covariant=True)
ProxyT = TypeVar("ProxyT", bound="DTypeProxy[DType]")

SupertypeRepr: TypeAlias = tuple[DType, DType, DType]
"""However we plan to represent a supertype relationship.

Means this, but without the nesting:

    (DType, DType) -> DType
"""


# TODO @dangotbanned: Special parametric
# TODO @dangotbanned: - Datetime
# TODO @dangotbanned: - Duration
# TODO @dangotbanned: - Array
# TODO @dangotbanned: - List
# TODO @dangotbanned: - Struct
# TODO @dangotbanned: Numeric
# TODO @dangotbanned: - _integer_supertyping
# TODO @dangotbanned: - _FLOAT_PROMOTE
# TODO @dangotbanned:   - (Decimal, Float32) -> Float64
# TODO @dangotbanned: - _primitive_numeric_supertyping
# TODO @dangotbanned:   - (Integer{32,64,128}, Float32) -> Float64
class Rule(Generic[D]):
    def iter_supertypes(self, owner: DTypeProxy[D], /) -> Iterator[SupertypeRepr]:
        raise NotImplementedError


@dataclass
class Unconditional(Rule[D]):
    others: tuple[DTypeProxy[DType], ...]

    def iter_supertypes(self, owner: DTypeProxy[D], /) -> Iterator[SupertypeRepr]:
        rights = chain.from_iterable(other.iter_instances() for other in self.others)
        for left, right in product(owner.iter_instances(), rights):
            yield left, right, left


@dataclass
class MatchSame(Rule[D]):
    predicate: Callable[[D, D], bool]

    def iter_supertypes(self, owner: DTypeProxy[D], /) -> Iterator[SupertypeRepr]:
        for left, right in product(owner.iter_instances(), owner.iter_instances()):
            if self.predicate(left, right):
                yield left, right, left


class _RuleNs(Generic[ProxyT, D]):
    def __init__(self, owner: ProxyT, /) -> None:
        self._owner: ProxyT = owner

    def unconditional(
        self, other: DTypeProxy[DType], *others: DTypeProxy[DType]
    ) -> ProxyT:
        """For all instances producable by `others`, this side is always a valid supertype."""
        return self._owner._with_rule(Unconditional((other, *others)))

    def match_same(self, predicate: Callable[[D, D], bool], /) -> ProxyT:
        """When comparing against an instance of the same type, `predicate` return True."""
        return self._owner._with_rule(MatchSame(predicate))


class DTypeProxy(Generic[D_co]):
    tp: type[D_co]
    _rules: tuple[Rule, ...] = ()

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{self.tp!r}]"

    def iter_instances(self) -> Iterator[D_co]:
        """Yield instances of the wrapped data type."""
        raise NotImplementedError

    def iter_supertypes(self) -> Iterator[SupertypeRepr]:
        """Yield all supertype relationships the wrapped data type is the winner of."""
        for rule in self.rules:
            yield from rule.iter_supertypes(self)

    def _with_rule(self, rule: Rule[Any], /) -> Self:
        """Extend the current rules, mutating this instance."""
        self._rules = *self._rules, rule
        return self

    @property
    def rule(self) -> _RuleNs[Self, D_co]:
        """Namespace for adding a supertyping rule for this DType.

        When a rule has mixed data types, it should be stored on the *"winning"* side of the comparison.
        """
        return _RuleNs(self)

    @property
    def rules(self) -> tuple[Rule, ...]:
        return self._rules


class Singleton(DTypeProxy[DS_co]):
    def __init__(self, tp: type[DS_co]) -> None:
        self.tp = tp

    def iter_instances(self) -> Iterator[DS_co]:
        yield self.instance

    @property
    def instance(self) -> DS_co:
        return self.tp()


class Parametric(DTypeProxy[DP_co]):
    params: _Params

    def __init__(self, tp: type[DP_co], params: _Params) -> None:
        self.tp = tp
        self.params = params

    def iter_instances(self) -> Iterator[DP_co]:
        sig = self._tp_signature()
        for bound in self.params.iter_bind_product(sig):
            yield self.tp(*bound.args, **bound.kwargs)

    def _tp_signature(self) -> inspect.Signature:
        return inspect.signature(self.tp)


# TODO @dangotbanned: Not sure exactly how this should work yet
# Maybe like selectors?
# Probably want more than a single nested proxy
class Nested(DTypeProxy[DN_co], Generic[DN_co, I_co]):
    inner: I_co

    def __init__(self, tp: type[DN_co], inner: I_co) -> None:
        self.tp = tp
        self.inner = inner

    def iter_instances(self) -> Iterator[DN_co]:
        for inner in self.inner.iter_instances():
            yield self.tp(inner)  # pyright: ignore[reportCallIssue, reportArgumentType]


_empty = _Params()


@overload
def dtype(tp: type[DS_co]) -> Singleton[DS_co]: ...
@overload
def dtype(tp: type[DP_co], params: _Params) -> Parametric[DP_co]: ...
def dtype(
    tp: type[DS_co | DP_co], params: _Params = _empty
) -> Singleton[DS_co] | Parametric[DP_co]:
    return (
        Parametric(tp, params)
        if issubclass(tp, (Datetime, Duration, Enum))
        else Singleton(tp)
    )


def arbitrary(*args: Any) -> Todo:
    """For `Datetime.time_zone`, it should be clear the exact values aren't important.

    It is the mismatch on comparison - so `arbitrary` represents an example to create an instance.
    """
    raise NotImplementedError


def describe_supertyping() -> Sequence[DTypeProxy[DType]]:  # noqa: PLR0914
    """Declarative form of supertyping rules.

    groups (per [polars/datatypes])

    [polars/datatypes]: https://docs.pola.rs/api/python/stable/reference/datatypes.html
    """
    # NOTE: Define each type, with enough content to derive all possible supertypes

    # String
    categorical = dtype(Categorical)
    enum = dtype(
        Enum, parametrize("categories", ["a", "b", "c"], ["d", "e", "f"])
    ).rule.match_same(lambda owner, other: owner.categories == other.categories)
    string = dtype(String).rule.unconditional(categorical, enum)
    group_string = (categorical, enum, string)

    # Other
    binary = dtype(Binary).rule.unconditional(string)
    boolean = dtype(Boolean)
    object_ = dtype(Object)
    unknown = dtype(Unknown)

    # Temporal
    date = dtype(Date)
    datetime = dtype(
        Datetime,
        parametrize("time_unit", "ns", "us", "ms", "s").parametrize(
            "time_zone", None, "UTC"
        ),
    ).rule.unconditional(date)
    duration = dtype(Duration, parametrize("time_unit", "ns", "us", "ms", "s"))
    time = dtype(Time)
    group_temporal = (date, datetime, duration, time)

    # Numeric
    i8 = dtype(Int8).rule.unconditional(boolean)
    i16 = dtype(Int16).rule.unconditional(boolean)
    i32 = dtype(Int32).rule.unconditional(boolean)
    i64 = dtype(Int64).rule.unconditional(boolean)
    i128 = dtype(Int128).rule.unconditional(boolean)
    u8 = dtype(UInt8).rule.unconditional(boolean)
    u16 = dtype(UInt16).rule.unconditional(boolean)
    u32 = dtype(UInt32).rule.unconditional(boolean)
    u64 = dtype(UInt64).rule.unconditional(boolean)
    u128 = dtype(UInt128).rule.unconditional(boolean)
    integer = i8, i16, i32, i64, i128, u8, u16, u32, u64, u128

    decimal = dtype(Decimal).rule.unconditional(boolean, *integer)
    f32 = dtype(Float32).rule.unconditional(boolean, i8, i16, u8, u16)
    f64 = dtype(Float64).rule.unconditional(boolean, f32, decimal, *integer)
    floating = f32, f64
    group_numeric = (decimal, *floating, *integer)

    # Nested
    # TODO @dangotbanned: Array + List can share some stuff, but mostly these three need pretty unique handling
    list_ = Nested(List, i64)
    # `Array.shape` rejection can use a rule like `Datetime.time_zone`
    array = Nested(Array, f64)
    struct = Nested(Struct, "todo!")  # pyright: ignore[reportArgumentType]
    group_nested = (list_, array, struct)

    group_known = (
        *group_string,
        *group_temporal,
        *group_numeric,
        *group_nested,
        binary,
        boolean,
        object_,
    )
    unknown = unknown.rule.unconditional(*group_known)
    return (*group_known, unknown)


def display_supertypes() -> None:
    proxies = describe_supertyping()
    # TODO @dangotbanned: Include Nested (and Unknown which refs them) when they're finished
    safe = (p for p in proxies if not (isinstance(p, Nested) or p.tp is Unknown))
    supertypes = chain.from_iterable(p.iter_supertypes() for p in safe)
    print(*supertypes, sep="\n")


if __name__ == "__main__":
    collect_supertypes()
