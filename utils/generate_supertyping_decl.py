from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib.util import find_spec
from itertools import chain, product
from typing import TYPE_CHECKING, Any, Generic, overload

import polars as pl

from narwhals._constants import MS_PER_SECOND, NS_PER_SECOND, US_PER_SECOND
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

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence
    from datetime import timezone

    from typing_extensions import Self, TypeAlias

    from narwhals.typing import TimeUnit

Todo: TypeAlias = Any
"""Placeholder type for planning interactions."""


Categories: TypeAlias = list[str]
TimeZone: TypeAlias = "str | timezone | None"
Values: TypeAlias = "Categories | TimeUnit | TimeZone"


def generate_supertypes_chart() -> Any:
    """Repro for [altair chart].

    [altair chart]: https://github.com/narwhals-dev/narwhals/pull/3396#discussion_r2702530394
    """
    if not find_spec("altair"):
        msg = (
            f"`{generate_supertypes_chart.__qualname__}()` requires `altair` to be installed.\n"
            "Hint: Try adding 'altair' to `[dependency-groups.docs]`"
        )
        raise ImportError(msg)
    import altair as alt

    from utils.generate_supertyping import _collect_supertypes

    orig_supertypes = _collect_supertypes()
    checkmark = pl.when("has_supertype").then(pl.lit("✔️")).otherwise(pl.lit("❌"))
    tooltip_label = (
        pl.when("has_supertype")
        .then(pl.format("({}, {}) -> {}", "left", "right", "supertype"))
        .otherwise("supertype")
    )
    df = (
        pl.DataFrame(orig_supertypes, schema=["left", "right", "supertype"], orient="row")
        .with_columns(has_supertype=pl.col("supertype") != pl.lit(""))
        .with_columns(has_supertype_repr=checkmark, rel=tooltip_label)
        .select("left", "right", "has_supertype_repr", "rel")
    )
    font_size = 14
    font = "'Lato', 'Segoe UI', Tahoma, Verdana, sans-serif"
    background = "#f5f5f5"  # color of code blocks

    base = alt.Chart(df).encode(
        alt.X("left").axis(orient="top"), alt.Y("right"), alt.Tooltip("rel").title(None)
    )
    chart = (
        base.mark_rect(filled=False) + base.mark_text().encode(text="has_supertype_repr")
    ).configure(
        axis=alt.theme.AxisConfigKwds(labelFontSize=font_size, title=None),
        mark=alt.theme.MarkConfigKwds(fontSize=font_size),
        background=background,
        font=font,
    )
    return chart  # noqa: RET504


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

_TIME_UNIT_PER_SECOND: Mapping[TimeUnit, int] = {
    "s": 1,
    "ms": MS_PER_SECOND,
    "us": US_PER_SECOND,
    "ns": NS_PER_SECOND,
}


def _key_fn_time_unit(obj: Datetime | Duration, /) -> int:
    return _TIME_UNIT_PER_SECOND[obj.time_unit]


SameTemporalT = TypeVar("SameTemporalT", Datetime, Duration)


def downcast_time_unit(
    left: SameTemporalT, right: SameTemporalT, /
) -> SameTemporalT | None:
    """Return the operand with the lowest precision time unit."""
    return min(left, right, key=_key_fn_time_unit)


# TODO @dangotbanned: Special parametric
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
    guard: Callable[[D, D], bool]

    def iter_supertypes(self, owner: DTypeProxy[D], /) -> Iterator[SupertypeRepr]:
        for left, right in product(owner.iter_instances(), owner.iter_instances()):
            if self.guard(left, right):
                yield left, right, left


@dataclass
class PromoteSame(Rule[D]):
    promotion: Callable[[D, D], DType | None]
    guard: Callable[[D, D], bool] | None

    def iter_supertypes(self, owner: DTypeProxy[D], /) -> Iterator[SupertypeRepr]:
        it: Iterator[tuple[D, D]] = product(
            owner.iter_instances(), owner.iter_instances()
        )
        if self.guard is not None:
            it = ((left, right) for left, right in it if self.guard(left, right))
        for left, right in it:
            if st := self.promotion(left, right):
                yield left, right, st


class _RuleNs(Generic[ProxyT, D]):
    def __init__(self, owner: ProxyT, /) -> None:
        self._owner: ProxyT = owner

    def unconditional(
        self, other: DTypeProxy[DType], *others: DTypeProxy[DType]
    ) -> ProxyT:
        """For all instances producable by `others`, this side is always a valid supertype."""
        return self._owner._with_rule(Unconditional((other, *others)))

    def match_same(self, guard: Callable[[D, D], bool], /) -> ProxyT:
        """When comparing against an instance of the same type, require `guard` for left to be a valid supertype."""
        return self._owner._with_rule(MatchSame(guard))

    def promote_same(
        self,
        promotion: Callable[[D, D], DType | None],
        guard: Callable[[D, D], bool] | None = None,
    ) -> ProxyT:
        """When comparing against an instance of the same type, use `promotion` to determine the supertype.

        Optionally first require `guard` to return True.
        """
        return self._owner._with_rule(PromoteSame(promotion, guard))


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
    datetime = (
        dtype(
            Datetime,
            parametrize("time_unit", "ns", "us", "ms", "s").parametrize(
                "time_zone", None, "UTC"
            ),
        )
        .rule.unconditional(date)
        .rule.promote_same(
            downcast_time_unit, lambda owner, other: owner.time_zone == other.time_zone
        )
    )
    duration = dtype(
        Duration, parametrize("time_unit", "ns", "us", "ms", "s")
    ).rule.promote_same(downcast_time_unit)
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
