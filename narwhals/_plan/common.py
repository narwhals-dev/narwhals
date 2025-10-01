from __future__ import annotations

import datetime as dt
import re
import sys
from collections.abc import Iterable
from decimal import Decimal
from operator import attrgetter
from secrets import token_hex
from typing import TYPE_CHECKING, cast, overload

from narwhals._plan._guards import is_iterable_reject
from narwhals._utils import _hasattr_static
from narwhals.dtypes import DType
from narwhals.exceptions import NarwhalsError
from narwhals.utils import Version

if TYPE_CHECKING:
    import reprlib
    from collections.abc import Iterator
    from typing import Any, Callable, ClassVar, TypeVar

    from typing_extensions import TypeIs

    from narwhals._plan.typing import (
        DTypeT,
        ExprIRT,
        FunctionT,
        NonNestedDTypeT,
        OneOrIterable,
    )
    from narwhals._utils import _StoresColumns
    from narwhals.typing import NonNestedDType, NonNestedLiteral

    T = TypeVar("T")


if sys.version_info >= (3, 13):
    from copy import replace as replace  # noqa: PLC0414
else:

    def replace(obj: T, /, **changes: Any) -> T:
        cls = obj.__class__
        func = getattr(cls, "__replace__", None)
        if func is None:
            msg = f"replace() does not support {cls.__name__} objects"
            raise TypeError(msg)
        return func(obj, **changes)  # type: ignore[no-any-return]


def pascal_to_snake_case(s: str) -> str:
    """Convert a PascalCase, camelCase string to snake_case.

    Adapted from https://github.com/pydantic/pydantic/blob/f7a9b73517afecf25bf898e3b5f591dffe669778/pydantic/alias_generators.py#L43-L62
    """
    # Handle the sequence of uppercase letters followed by a lowercase letter
    snake = _PATTERN_UPPER_LOWER.sub(_re_repl_snake, s)
    # Insert an underscore between a lowercase letter and an uppercase letter
    return _PATTERN_LOWER_UPPER.sub(_re_repl_snake, snake).lower()


_PATTERN_UPPER_LOWER = re.compile(r"([A-Z]+)([A-Z][a-z])")
_PATTERN_LOWER_UPPER = re.compile(r"([a-z])([A-Z])")


def _re_repl_snake(match: re.Match[str], /) -> str:
    return f"{match.group(1)}_{match.group(2)}"


def dispatch_method_name(tp: type[ExprIRT | FunctionT]) -> str:
    config = tp.__expr_ir_config__
    name = config.override_name or pascal_to_snake_case(tp.__name__)
    return f"{ns}.{name}" if (ns := getattr(config, "accessor_name", "")) else name


def dispatch_getter(tp: type[ExprIRT | FunctionT]) -> Callable[[Any], Any]:
    getter = attrgetter(dispatch_method_name(tp))
    if tp.__expr_ir_config__.origin == "expr":
        return getter
    return lambda ctx: getter(ctx.__narwhals_namespace__())


def py_to_narwhals_dtype(obj: NonNestedLiteral, version: Version = Version.MAIN) -> DType:
    dtypes = version.dtypes
    mapping: dict[type[NonNestedLiteral], type[NonNestedDType]] = {
        int: dtypes.Int64,
        float: dtypes.Float64,
        str: dtypes.String,
        bool: dtypes.Boolean,
        dt.datetime: dtypes.Datetime,
        dt.date: dtypes.Date,
        dt.time: dtypes.Time,
        dt.timedelta: dtypes.Duration,
        bytes: dtypes.Binary,
        Decimal: dtypes.Decimal,
        type(None): dtypes.Unknown,
    }
    return mapping.get(type(obj), dtypes.Unknown)()


@overload
def into_dtype(dtype: type[NonNestedDTypeT], /) -> NonNestedDTypeT: ...
@overload
def into_dtype(dtype: DTypeT, /) -> DTypeT: ...
def into_dtype(dtype: DTypeT | type[NonNestedDTypeT], /) -> DTypeT | NonNestedDTypeT:
    # NOTE: `mypy` needs to learn intersections
    if isinstance(dtype, type) and issubclass(dtype, DType):
        return cast("NonNestedDTypeT", dtype())
    return dtype


# TODO @dangotbanned: Review again and try to work around (https://github.com/microsoft/pyright/issues/10673#issuecomment-3033789021)
# The issue is `T` possibly being `Iterable`
# Ignoring here still leaks the issue to the caller, where you need to annotate the base case
def flatten_hash_safe(iterable: Iterable[OneOrIterable[T]], /) -> Iterator[T]:
    """Fully unwrap all levels of nesting.

    Aiming to reduce the chances of passing an unhashable argument.
    """
    for element in iterable:
        if isinstance(element, Iterable) and not is_iterable_reject(element):
            yield from flatten_hash_safe(element)
        else:
            yield element  # type: ignore[misc]


def _has_columns(obj: Any) -> TypeIs[_StoresColumns]:
    return _hasattr_static(obj, "columns")


def _reprlib_repr_backport() -> reprlib.Repr:
    # 3.12 added `indent` https://github.com/python/cpython/issues/92734
    # but also a useful constructor https://github.com/python/cpython/issues/94343
    import reprlib

    if sys.version_info >= (3, 12):
        return reprlib.Repr(indent=4, maxlist=10)
    else:  # pragma: no cover  # noqa: RET505
        obj = reprlib.Repr()
        obj.maxlist = 10
        return obj


class temp:  # noqa: N801
    """Temporary mini namespace for temporary utils."""

    _MAX_ITERATIONS: ClassVar[int] = 100
    _MIN_RANDOM_CHARS: ClassVar[int] = 4

    @classmethod
    def column_name(
        cls,
        source: _StoresColumns | Iterable[str],
        /,
        *,
        prefix: str = "nw",
        n_chars: int = 16,
    ) -> str:
        """Generate a single, unique column name that is not present in `source`.

        Arguments:
            source: Source of columns to check for uniqueness.
            prefix: Prepend the name with this string.
            n_chars: Total number of characters used by the name (including `prefix`).

        Examples:
            >>> import narwhals as nw
            >>> from narwhals._plan.common import temp
            >>> columns = "abc", "xyz"
            >>> temp.column_name(columns)  # doctest: +SKIP
            'nwf65daf7ceb3c2f'

            Limit the number of characters that the name uses

            >>> temp.column_name(columns, n_chars=8)  # doctest: +SKIP
            'nw388b5d'

            Make the name easier to trace back

            >>> temp.column_name(columns, prefix="_its_a_me_")  # doctest: +SKIP
            '_its_a_me_0ea2b0'

            Pass in a `DataFrame` directly, and let us get the columns for you

            >>> df = nw.from_dict({"foo": [1, 2], "bar": [6.0, 7.0]}, backend="polars")
            >>> df.with_row_index(temp.column_name(df, prefix="idx_"))  # doctest: +SKIP
            ┌────────────────────────────────┐
            |       Narwhals DataFrame       |
            |--------------------------------|
            |shape: (2, 3)                   |
            |┌──────────────────┬─────┬─────┐|
            |│ idx_bae5e1b22963 ┆ foo ┆ bar │|
            |│ ---              ┆ --- ┆ --- │|
            |│ u32              ┆ i64 ┆ f64 │|
            |╞══════════════════╪═════╪═════╡|
            |│ 0                ┆ 1   ┆ 6.0 │|
            |│ 1                ┆ 2   ┆ 7.0 │|
            |└──────────────────┴─────┴─────┘|
            └────────────────────────────────┘
        """
        columns = cls._into_columns(source)
        prefix, n_bytes = cls._parse_prefix_n_bytes(prefix, n_chars)
        for _ in range(cls._MAX_ITERATIONS):
            token = f"{prefix}{token_hex(n_bytes)}"
            if token not in columns:
                return token
        raise cls._failed_generation_error(columns, n_chars)

    # TODO @dangotbanned: Write examples
    @classmethod
    def column_names(
        cls,
        source: _StoresColumns | Iterable[str],
        /,
        *,
        prefix: str = "nw",
        n_chars: int = 16,
    ) -> Iterator[str]:
        """Yields unique column names that are not present in `source`.

        Any column name returned will be unique among those that preceded it.

        Arguments:
            source: Source of columns to check for uniqueness.
            prefix: Prepend the name with this string.
            n_chars: Total number of characters used by the name (including `prefix`).
        """
        columns = cls._into_columns(source)
        prefix, n_bytes = cls._parse_prefix_n_bytes(prefix, n_chars)
        n_failed: int = 0
        while n_failed <= cls._MAX_ITERATIONS:
            token = f"{prefix}{token_hex(n_bytes)}"
            if token not in columns:
                columns.add(token)
                n_failed = 0
                yield token
            else:
                n_failed += 1
        raise cls._failed_generation_error(columns, n_chars)

    @staticmethod
    def _into_columns(source: _StoresColumns | Iterable[str], /) -> set[str]:
        return set(source.columns if _has_columns(source) else source)

    @classmethod
    def _parse_prefix_n_bytes(cls, prefix: str, n_chars: int, /) -> tuple[str, int]:
        prefix = prefix or "nw"
        if not (available := n_chars - len(prefix)) or available < cls._MIN_RANDOM_CHARS:
            raise cls._not_enough_room_error(prefix, n_chars)
        return prefix, available // 2

    @classmethod
    def _not_enough_room_error(cls, prefix: str, n_chars: int, /) -> NarwhalsError:
        len_prefix = len(prefix)
        available_chars = n_chars - len_prefix
        if available_chars < 0:
            visualize = ""
        else:
            okay = "✔" * available_chars
            bad = "✖" * (cls._MIN_RANDOM_CHARS - available_chars)
            visualize = f"\n    Preview: '{prefix}{okay}{bad}'"
        msg = (
            f"Temporary column name generation requires {len_prefix} characters for the prefix "
            f"and at least {cls._MIN_RANDOM_CHARS} more to store random bytes:{visualize}\n\n"
            f"Hint: Maybe try\n"
            f"- a shorter `prefix` than {prefix!r}?\n"
            f"- a higher `n_chars` than {n_chars!r}?"
        )
        return NarwhalsError(msg)

    @classmethod
    def _failed_generation_error(
        cls, columns: Iterable[str], n_chars: int, /
    ) -> NarwhalsError:
        current = sorted(columns)
        truncated = _reprlib_repr_backport().repr(current)
        msg = (
            "Was unable to generate a column name with "
            f"`{n_chars=}` within {cls._MAX_ITERATIONS} iterations, \n"
            f"that was not present in existing ({len(current)}) columns:\n{truncated}"
        )
        return NarwhalsError(msg)
