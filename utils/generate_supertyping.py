"""Adapted from [@FBruzzesi script (2026-01-11)].

[@FBruzzesi script (2026-01-11)]: https://github.com/narwhals-dev/narwhals/pull/3396#issuecomment-3733465005
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Generic, Literal, TypeVar, overload

import polars as pl

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
    from collections.abc import Iterator
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

D_co = TypeVar("D_co", bound=DType, covariant=True)
DS_co = TypeVar("DS_co", bound=_Singleton, covariant=True)
DP_co = TypeVar("DP_co", bound=_Parametric, covariant=True)
DN_co = TypeVar("DN_co", bound=_Nested, covariant=True)
I_co = TypeVar("I_co", bound="DTypeProxy[DType]", covariant=True)
ProxyT = TypeVar("ProxyT", bound="DTypeProxy[DType]")


RuleKind: TypeAlias = Literal["unconditional"]

WipRuleRepr: TypeAlias = "tuple[RuleKind, tuple[DTypeProxy[DType], ...]]"
"""Will do for now, but more structure might be nice."""


class _RuleNs(Generic[ProxyT]):
    def __init__(self, owner: ProxyT, /) -> None:
        self._owner: ProxyT = owner

    def unconditional(
        self, other: DTypeProxy[DType], *others: DTypeProxy[DType]
    ) -> ProxyT:
        """For all instances producable by `others`, this side is always a valid supertype."""
        rule_repr: WipRuleRepr = ("unconditional", (other, *others))
        return self._owner._with_rule(rule_repr)


class DTypeProxy(Generic[D_co]):
    tp: type[D_co]
    _rules: tuple[WipRuleRepr, ...] = ()

    def iter_instances(self) -> Iterator[D_co]:
        raise NotImplementedError

    def _with_rule(self, rule: Any, /) -> Self:
        """Extend the current rules, mutating this instance."""
        self._rules = *self._rules, rule
        return self

    @property
    def rule(self) -> _RuleNs[Self]:
        """Namespace for adding a supertyping rule for this DType.

        When a rule has mixed data types, it should be stored on the *"winning"* side of the comparison.
        """
        return _RuleNs(self)

    @property
    def rules(self) -> tuple[WipRuleRepr, ...]:
        return self._rules


class Singleton(DTypeProxy[DS_co]):
    def __init__(self, tp: type[DS_co]) -> None:
        self.tp = tp

    def iter_instances(self) -> Iterator[DS_co]:
        yield self.instance

    @property
    def instance(self) -> DS_co:
        return self.tp()

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{self.tp!r}]"


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


def describe_supertyping() -> None:  # noqa: PLR0914
    """Declarative form of supertyping rules.

    groups (per [polars/datatypes])

    [polars/datatypes]: https://docs.pola.rs/api/python/stable/reference/datatypes.html
    """
    # NOTE: Define each type, with enough content to derive all possible supertypes

    # String
    categorical = dtype(Categorical)
    enum = dtype(Enum, parametrize("categories", ["a", "b", "c"], ["d", "e", "f"]))
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
    decimal = dtype(Decimal).rule.unconditional(boolean)
    f32 = dtype(Float32).rule.unconditional(boolean)
    f64 = dtype(Float64).rule.unconditional(boolean)
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
    group_numeric = (decimal, f32, f64, i8, i16, i32, i64, i128, u8, u16, u32, u64, u128)

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


if __name__ == "__main__":
    collect_supertypes()
