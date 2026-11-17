"""Helpers vendored from Narwhals' private modules.

`narwhals-dict` only imports from Narwhals' public API, so anything it needs from
`narwhals._*` is copied here instead. Keep these in sync with upstream on a
best-effort basis, there is no requirement that they stay identical.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Sequence
from itertools import chain
from typing import TYPE_CHECKING, Any, Generic, Literal, NoReturn, TypeVar, cast

from narwhals.exceptions import DuplicateError
from narwhals.utils import Implementation

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Container, Iterable, Mapping
    from typing import TypeAlias

    from typing_extensions import TypeIs

    from narwhals.typing import SizeUnit
    from narwhals.utils import Version

_T = TypeVar("_T")
FrameT = TypeVar("FrameT")
SeriesT = TypeVar("SeriesT")

EvalNames: TypeAlias = "Callable[[FrameT], Sequence[str]]"
EvalSeries: TypeAlias = "Callable[[FrameT], Sequence[SeriesT]]"
AliasNames: TypeAlias = "Callable[[Sequence[str]], Sequence[str]]"
NarwhalsAggregation: TypeAlias = Literal[
    "sum",
    "mean",
    "median",
    "max",
    "min",
    "mode",
    "std",
    "var",
    "len",
    "n_unique",
    "count",
    "quantile",
    "all",
    "any",
    "first",
    "last",
    "any_value",
]


def assert_never(arg: Any, /) -> NoReturn:  # pragma: no cover
    msg = f"Expected code to be unreachable, but got: {arg!r}"
    raise AssertionError(msg)


# --- `narwhals._utils` ---------------------------------------------------------------


class not_implemented:  # noqa: N801
    """Mark some functionality as unsupported, raising `NotImplementedError` on access."""

    def __init__(self, alias: str | None = None, /, *, hint: str = "") -> None:
        self._alias: str | None = alias
        self._hint: str = hint

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>: {self._name_owner}.{self._name}"

    def __set_name__(self, owner: type[Any], name: str) -> None:
        self._name_owner: str = owner.__name__
        self._name: str = self._alias or name

    def __get__(self, instance: Any, owner: type[Any] | None = None, /) -> Any:
        if instance is None:
            return self
        implementation = getattr(instance, "_implementation", Implementation.UNKNOWN)
        who = (
            repr(implementation)
            if implementation is not Implementation.UNKNOWN
            else self._name_owner
        )
        fallback_msg = (
            "If you would like to see this functionality in `narwhals`, "
            "please open an issue at: https://github.com/narwhals-dev/narwhals/issues"
        )
        msg = f"{self._name!r} is not implemented for: {who!r}.\n\n{self._hint or fallback_msg}"
        raise NotImplementedError(msg)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.__get__("raise")


def check_column_names_are_unique(columns: Collection[str]) -> None:
    if len(columns) != len(set(columns)):
        from collections import Counter

        counter = Counter(columns)
        duplicates = {k: v for k, v in counter.items() if v > 1}
        msg = "".join(f"\n- '{k}' {v} times" for k, v in duplicates.items())
        msg = f"Expected unique column names, got:{msg}"
        raise DuplicateError(msg)


def is_sequence_of(obj: Any, tp: type[_T]) -> TypeIs[Sequence[_T]]:
    # Only sniffs the first element.
    return bool(
        isinstance(obj, Sequence)
        and not isinstance(obj, str)
        and (first := next(iter(obj), None))
        and isinstance(first, tp)
    )


def isinstance_or_issubclass(obj_or_cls: Any, cls_or_tuple: Any) -> bool:
    from narwhals.dtypes import DType

    if isinstance(obj_or_cls, DType):
        return isinstance(obj_or_cls, cls_or_tuple)
    return isinstance(obj_or_cls, cls_or_tuple) or (
        isinstance(obj_or_cls, type) and issubclass(obj_or_cls, cls_or_tuple)
    )


def scale_bytes(sz: int, unit: SizeUnit) -> int | float:
    if unit in {"b", "bytes"}:
        return sz
    if unit in {"kb", "kilobytes"}:
        return sz / 1024
    if unit in {"mb", "megabytes"}:
        return sz / 1024**2
    if unit in {"gb", "gigabytes"}:
        return sz / 1024**3
    if unit in {"tb", "terabytes"}:
        return sz / 1024**4
    msg = f"`unit` must be one of {{'b', 'kb', 'mb', 'gb', 'tb'}}, got {unit!r}"
    raise ValueError(msg)


# --- `narwhals._expression_parsing` --------------------------------------------------


def combine_evaluate_output_names(*exprs: Any) -> EvalNames[Any]:
    # Follow left-hand-rule for naming. E.g. `nw.sum_horizontal(expr1, expr2)` takes the
    # first name of `expr1`.
    def evaluate_output_names(df: Any) -> Sequence[str]:
        return exprs[0]._evaluate_output_names(df)[:1]

    return evaluate_output_names


def combine_alias_output_names(*exprs: Any) -> AliasNames | None:
    if exprs[0]._alias_output_names is None:
        return None

    def alias_output_names(names: Sequence[str]) -> Sequence[str]:
        return exprs[0]._alias_output_names(names)[:1]

    return alias_output_names


def evaluate_output_names_and_aliases(
    expr: Any, df: Any, exclude: Sequence[str]
) -> tuple[Sequence[str], Sequence[str]]:
    output_names: Sequence[str] = expr._evaluate_output_names(df)
    aliases: Sequence[str] = (
        alias_fn(output_names)
        if (alias_fn := expr._alias_output_names) is not None
        else output_names
    )
    if exclude and expr._metadata.expansion_kind.is_multi_unnamed():
        exclude_set = frozenset(exclude)
        output_names, aliases = zip(
            *[
                (x, alias)
                for x, alias in zip(output_names, aliases, strict=True)
                if x not in exclude_set
            ],
            strict=True,
        )
    return output_names, aliases


def evaluate_aliases(frame: Any, exprs: Iterable[Any], /) -> list[str]:
    return list(chain.from_iterable(expr._evaluate_aliases(frame) for expr in exprs))


# --- `narwhals._compliant` -----------------------------------------------------------


class SeriesNamespace(Generic[SeriesT]):
    """Base for `Series.<accessor>` namespaces, wrapping a compliant series."""

    def __init__(self, series: SeriesT, /) -> None:
        self._compliant_series = series

    @property
    def compliant(self) -> SeriesT:
        return self._compliant_series

    @property
    def implementation(self) -> Implementation:
        return self._compliant_series._implementation  # type: ignore[attr-defined]

    @property
    def backend_version(self) -> tuple[int, ...]:
        return self.implementation._backend_version()

    @property
    def native(self) -> Any:
        return self._compliant_series.native  # type: ignore[attr-defined]

    @property
    def version(self) -> Version:
        return self._compliant_series._version  # type: ignore[attr-defined]

    def with_native(self, series: Any, /) -> SeriesT:
        return self._compliant_series._with_native(series)  # type: ignore[attr-defined]


# --- `narwhals._duration` ------------------------------------------------------------

IntervalUnit: TypeAlias = Literal["ns", "us", "ms", "s", "m", "h", "d", "mo", "q", "y"]
_PATTERN_INTERVAL: re.Pattern[str] = re.compile(
    r"^(?P<multiple>-?\d+)(?P<unit>ns|us|ms|mo|m|s|h|d|q|y)\Z"
)
_MONTH_MULTIPLES = frozenset([1, 2, 3, 4, 6, 12])
_QUARTER_MULTIPLES = frozenset([1, 2, 4])
_UNIT_TO_TIMEDELTA: Mapping[IntervalUnit, str] = {
    "d": "days",
    "h": "hours",
    "m": "minutes",
    "s": "seconds",
    "ms": "milliseconds",
    "us": "microseconds",
}


class Interval:
    def __init__(self, multiple: int, unit: IntervalUnit, /) -> None:
        self.multiple: int = multiple
        self.unit: IntervalUnit = unit

    def to_timedelta(
        self, *, unsupported: Container[IntervalUnit] = frozenset(("ns", "mo", "q", "y"))
    ) -> dt.timedelta:
        if self.unit in unsupported:  # pragma: no cover
            msg = f"Creating timedelta with {self.unit} unit is not supported."
            raise NotImplementedError(msg)
        kwd = _UNIT_TO_TIMEDELTA[self.unit]
        return dt.timedelta(**{kwd: self.multiple})

    @classmethod
    def parse(cls, every: str) -> Interval:
        multiple, unit = cls._parse(every)
        if unit == "mo" and multiple not in _MONTH_MULTIPLES:
            msg = f"Only the following multiples are supported for 'mo' unit: {_MONTH_MULTIPLES}.\nGot: {multiple}."
            raise ValueError(msg)
        if unit == "q" and multiple not in _QUARTER_MULTIPLES:
            msg = f"Only the following multiples are supported for 'q' unit: {_QUARTER_MULTIPLES}.\nGot: {multiple}."
            raise ValueError(msg)
        if unit == "y" and multiple != 1:
            msg = (
                f"Only multiple 1 is currently supported for 'y' unit.\nGot: {multiple}."
            )
            raise ValueError(msg)
        return cls(multiple, unit)

    @classmethod
    def parse_no_constraints(cls, every: str) -> Interval:
        return cls(*cls._parse(every))

    @staticmethod
    def _parse(every: str) -> tuple[int, IntervalUnit]:
        from typing import get_args

        if match := _PATTERN_INTERVAL.match(every):
            return int(match["multiple"]), cast("IntervalUnit", match["unit"])
        msg = (
            f"Invalid `every` string: {every}. Expected string of kind <number><unit>, "
            f"where 'unit' is one of: {get_args(IntervalUnit)}."
        )
        raise ValueError(msg)


# --- `narwhals._arrow.utils` (format inference for `str.to_datetime`) ---------------

DATE_RE = r"(?P<date>\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{8})"
SEP_RE = r"(?P<sep>\s|T)"
TIME_RE = r"(?P<time>\d{2}:\d{2}(?::\d{2})?|\d{6}?)"
HMS_RE = r"^(?P<hms>\d{2}:\d{2}:\d{2})$"
HM_RE = r"^(?P<hm>\d{2}:\d{2})$"
HMS_RE_NO_SEP = r"^(?P<hms_no_sep>\d{6})$"
TZ_RE = r"(?P<tz>Z|[+-]\d{2}:?\d{2})"  # 'Z', '+02:00', '+0200'
FULL_RE = rf"{DATE_RE}{SEP_RE}?{TIME_RE}?{TZ_RE}?$"

YMD_RE = r"^(?P<year>(?:[12][0-9])?[0-9]{2})(?P<sep1>[-/.])(?P<month>0[1-9]|1[0-2])(?P<sep2>[-/.])(?P<day>0[1-9]|[12][0-9]|3[01])$"
DMY_RE = r"^(?P<day>0[1-9]|[12][0-9]|3[01])(?P<sep1>[-/.])(?P<month>0[1-9]|1[0-2])(?P<sep2>[-/.])(?P<year>(?:[12][0-9])?[0-9]{2})$"
MDY_RE = r"^(?P<month>0[1-9]|1[0-2])(?P<sep1>[-/.])(?P<day>0[1-9]|[12][0-9]|3[01])(?P<sep2>[-/.])(?P<year>(?:[12][0-9])?[0-9]{2})$"
YMD_RE_NO_SEP = r"^(?P<year>(?:[12][0-9])?[0-9]{2})(?P<month>0[1-9]|1[0-2])(?P<day>0[1-9]|[12][0-9]|3[01])$"

DATE_FORMATS = (
    (YMD_RE_NO_SEP, "%Y%m%d"),
    (YMD_RE, "%Y-%m-%d"),
    (DMY_RE, "%d-%m-%Y"),
    (MDY_RE, "%m-%d-%Y"),
)
TIME_FORMATS = ((HMS_RE, "%H:%M:%S"), (HM_RE, "%H:%M"), (HMS_RE_NO_SEP, "%H%M%S"))
