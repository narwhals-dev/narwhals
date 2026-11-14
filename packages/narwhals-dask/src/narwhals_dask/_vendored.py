"""Helpers vendored from Narwhals' private modules.

`narwhals-dask` only imports from Narwhals' public API, so anything it needs from
`narwhals._*` is copied here instead. Keep these in sync with upstream on a
best-effort basis, there is no requirement that they stay identical.
"""

from __future__ import annotations

import datetime as dt
import operator
import re
from itertools import chain
from typing import TYPE_CHECKING, Any, Generic, Literal, NoReturn, TypeVar, cast

from narwhals import Implementation
from narwhals.exceptions import ColumnNotFoundError, DuplicateError

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Collection,
        Container,
        Iterable,
        Mapping,
        Sequence,
    )
    from typing import TypeAlias

    import pandas as pd
    from typing_extensions import TypeIs

    from narwhals.typing import (
        DTypeBackend,
        NormalizedPath,
        NormalizedSource,
        PythonLiteral,
        TimeUnit,
    )

_T = TypeVar("_T")
FrameT = TypeVar("FrameT")
SeriesT = TypeVar("SeriesT")
ExprT = TypeVar("ExprT")

EvalNames: TypeAlias = "Callable[[FrameT], Sequence[str]]"
EvalSeries: TypeAlias = "Callable[[FrameT], Sequence[SeriesT]]"
AliasName: TypeAlias = "Callable[[str], str]"
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

# Temporal constants (from `narwhals._constants`).
SECONDS_PER_DAY = 86_400
NS_PER_SECOND = 1_000_000_000
US_PER_SECOND = 1_000_000
MS_PER_SECOND = 1_000
NS_PER_MICROSECOND = 1_000
NS_PER_MILLISECOND = 1_000_000


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


def length_changing_hint(*, instead_of: str, use: str) -> str:
    return (
        "Length-changing expressions (e.g. `unique`, `drop_nulls`, `filter`) are not\n"
        "supported for this backend, not even when followed by an aggregation.\n\n"
        f"Hint: instead of `{instead_of}`, use `{use}`."
    )


def check_columns_exist(
    subset: Collection[str], /, *, available: Collection[str]
) -> ColumnNotFoundError | None:
    if missing := set(subset).difference(available):
        return ColumnNotFoundError.from_missing_and_available_column_names(
            missing, available
        )
    return None


def check_column_names_are_unique(columns: Collection[str]) -> None:
    if len(columns) != len(set(columns)):
        from collections import Counter

        counter = Counter(columns)
        duplicates = {k: v for k, v in counter.items() if v > 1}
        msg = "".join(f"\n- '{k}' {v} times" for k, v in duplicates.items())
        msg = f"Expected unique column names, got:{msg}"
        raise DuplicateError(msg)


def parse_columns_to_drop(
    columns: Sequence[str], subset: Iterable[str], /, *, strict: bool
) -> list[str]:
    if not strict:
        return list(set(columns).intersection(subset))
    to_drop = list(subset)
    if error := check_columns_exist(to_drop, available=columns):
        raise error
    return to_drop


def remap_full_join_keys(
    left_on: Collection[str], right_on: Collection[str], suffix: str
) -> dict[str, str]:
    """Remap right join keys, appending `suffix` to those which collide with left keys."""
    right_keys_suffixed = (
        f"{key}{suffix}" if key in left_on else key for key in right_on
    )
    return dict(zip(right_on, right_keys_suffixed, strict=False))


def is_sequence_of(obj: Any, tp: type[_T]) -> TypeIs[Sequence[_T]]:
    from collections.abc import Sequence

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


def str_slice_stop(offset: int, length: int | None) -> int | None:
    """Translate Polars' `str.slice(offset, length)` into a Python-slice `stop`."""
    if length is None or offset < 0 <= offset + length:
        return None
    return offset + length


def validate_separators(
    separator: str, native_separators: tuple[str, ...], kwds: Mapping[str, Any], /
) -> None:
    for native_separator in native_separators:
        if native_separator in kwds and kwds[native_separator] != separator:
            msg = (
                f"`separator` and `{native_separator}` do not match: "
                f"`separator`={separator} and `{native_separator}`={kwds[native_separator]}."
            )
            raise TypeError(msg)


def ensure_path_source(source: NormalizedSource, /) -> NormalizedPath:
    """Reject file-like objects, `dask` readers require a path."""
    from io import IOBase

    if isinstance(source, IOBase):
        msg = (
            f"Reading from a file-like object is not supported for the {Implementation.DASK} backend.\n\n"
            "Hint: use 'pandas', 'polars' or 'pyarrow', or write the buffer to a file first."
        )
        raise TypeError(msg)
    return cast("NormalizedPath", source)


# --- `narwhals._expression_parsing` --------------------------------------------------


def combine_evaluate_output_names(*exprs: Any) -> EvalNames[Any]:
    # Follow left-hand-rule for naming. E.g. `nw.sum_horizontal(expr1, expr2)` takes the
    # first name of `expr1`.
    def evaluate_output_names(df: Any) -> Sequence[str]:
        return exprs[0]._evaluate_output_names(df)[:1]  # type: ignore[no-any-return]

    return evaluate_output_names


def combine_alias_output_names(*exprs: Any) -> AliasNames | None:
    if exprs[0]._alias_output_names is None:
        return None

    def alias_output_names(names: Sequence[str]) -> Sequence[str]:
        return exprs[0]._alias_output_names(names)[:1]  # type: ignore[no-any-return]

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


class ExprNamespace(Generic[ExprT]):
    """Base for `Expr.<accessor>` namespaces, wrapping a compliant expression."""

    def __init__(self, expr: ExprT, /) -> None:
        self._compliant_expr = expr

    @property
    def compliant(self) -> ExprT:
        return self._compliant_expr


# --- `narwhals._pandas_like` ---------------------------------------------------------

ALIAS_DICT = {"d": "D", "m": "min"}

PANDAS_REMAP_AGGS: Mapping[NarwhalsAggregation, str] = {
    "sum": "sum",
    "mean": "mean",
    "median": "median",
    "max": "max",
    "min": "min",
    "mode": "mode",
    "std": "std",
    "var": "var",
    "len": "size",
    "n_unique": "nunique",
    "count": "count",
    "quantile": "quantile",
    "all": "all",
    "any": "any",
    "first": "nth",
    "last": "nth",
    "any_value": "nth",
}


def make_group_by_kwargs(*, drop_null_keys: bool) -> dict[str, bool]:
    # NOTE: `as_index` is not supported in dask
    return {"sort": False, "dropna": drop_null_keys, "observed": True}


def is_dtype_pyarrow(dtype: Any) -> TypeIs[pd.ArrowDtype]:
    import pandas as pd

    return hasattr(pd, "ArrowDtype") and isinstance(dtype, pd.ArrowDtype)


def is_dtype_numpy_nullable(dtype: Any) -> bool:
    import pandas as pd

    # NOTE: We need a sentinel as the positive case is `BaseMaskedDtype.base = None`
    sentinel = object()
    return (
        isinstance(dtype, pd.api.extensions.ExtensionDtype)
        and getattr(dtype, "base", sentinel) is None
    )


def get_dtype_backend(dtype: Any) -> DTypeBackend:
    if is_dtype_pyarrow(dtype):
        return "pyarrow"
    return "numpy_nullable" if is_dtype_numpy_nullable(dtype) else None


def select_columns_by_name(df: Any, column_names: list[str]) -> Any:
    """Select columns by name, prefer this over `df.loc[:, column_names]`."""
    if len(column_names) == len(df.columns) and all(
        x == y for x, y in zip(df.columns, column_names, strict=True)
    ):
        return df
    if df.columns.dtype.kind == "b":
        # See https://github.com/narwhals-dev/narwhals/issues/1349#issuecomment-2470118122
        if error := check_columns_exist(column_names, available=df.columns.tolist()):
            raise error
        return df.loc[:, column_names]
    try:
        return df[column_names]
    except KeyError as e:
        if error := check_columns_exist(column_names, available=df.columns.tolist()):
            raise error from e
        raise


_TIMESTAMP_DATETIME_OP_FACTOR: Mapping[
    tuple[TimeUnit, TimeUnit], tuple[Callable[[Any, int], Any], int]
] = {
    ("ns", "us"): (operator.floordiv, 1_000),
    ("ns", "ms"): (operator.floordiv, 1_000_000),
    ("us", "ns"): (operator.mul, NS_PER_MICROSECOND),
    ("us", "ms"): (operator.floordiv, 1_000),
    ("ms", "ns"): (operator.mul, NS_PER_MILLISECOND),
    ("ms", "us"): (operator.mul, 1_000),
    ("s", "ns"): (operator.mul, NS_PER_SECOND),
    ("s", "us"): (operator.mul, US_PER_SECOND),
    ("s", "ms"): (operator.mul, MS_PER_SECOND),
}

_TIMESTAMP_DATE_FACTOR: Mapping[TimeUnit, int] = {
    "ns": NS_PER_SECOND,
    "us": US_PER_SECOND,
    "ms": MS_PER_SECOND,
    "s": 1,
}


def calculate_timestamp_datetime(s: Any, current: TimeUnit, time_unit: TimeUnit) -> Any:
    if current == time_unit:
        return s
    if item := _TIMESTAMP_DATETIME_OP_FACTOR.get((current, time_unit)):
        fn, factor = item
        return fn(s, factor)
    msg = f"unexpected time unit {current}, please report an issue at https://github.com/narwhals-dev/narwhals"  # pragma: no cover
    raise AssertionError(msg)  # pragma: no cover


def calculate_timestamp_date(s: Any, time_unit: TimeUnit) -> Any:
    return s * SECONDS_PER_DAY * _TIMESTAMP_DATE_FACTOR[time_unit]


def window_kwargs_to_pandas_equivalent(
    function_name: str, kwargs: dict[str, Any]
) -> dict[str, PythonLiteral]:
    """Translate the kwargs of an aggregation in `over` context to pandas `transform` ones.

    Only covers the aggregations which Dask supports in `over`, see `PANDAS_REMAP_AGGS`.
    """
    if function_name == "n_unique":
        return {"dropna": False}
    if function_name in {"std", "var"}:
        return {"ddof": kwargs["ddof"]}
    if function_name == "quantile":
        return {"q": kwargs["quantile"], "interpolation": kwargs["interpolation"]}
    return {}


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
