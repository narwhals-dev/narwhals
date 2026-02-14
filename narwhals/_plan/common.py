from __future__ import annotations

import datetime as dt
import sys
from collections.abc import Iterable
from decimal import Decimal
from io import BytesIO
from secrets import token_hex
from typing import TYPE_CHECKING, cast, overload

from narwhals._plan._guards import is_iterable_reject
from narwhals._utils import _hasattr_static, qualified_type_name
from narwhals.dtypes import DType
from narwhals.exceptions import NarwhalsError
from narwhals.utils import Version

if TYPE_CHECKING:
    import reprlib
    from collections.abc import Iterator
    from typing import Any, ClassVar, TypeVar

    from typing_extensions import TypeIs

    from narwhals._plan.compliant.series import CompliantSeries
    from narwhals._plan.series import Series
    from narwhals._plan.typing import (
        ColumnNameOrSelector,
        DTypeT,
        NonNestedDTypeT,
        OneOrIterable,
        Seq,
    )
    from narwhals._utils import _StoresColumns
    from narwhals.typing import FileSource, NonNestedDType, NonNestedLiteral

    T = TypeVar("T")


if sys.version_info >= (3, 13):  # pragma: no cover
    from copy import replace as replace  # noqa: PLC0414
else:  # pragma: no cover

    def replace(obj: T, /, **changes: Any) -> T:
        cls = obj.__class__
        func = getattr(cls, "__replace__", None)
        if func is None:
            msg = f"replace() does not support {cls.__name__} objects"
            raise TypeError(msg)
        return func(obj, **changes)  # type: ignore[no-any-return]


IDX_DTYPE = Version.MAIN.dtypes.Int64()
"""TODO @dangotbanned: Unify `IDX_DTYPE` as backends are mixed:

- UInt32 ([polars] excluding `bigidx`)
- UInt64 ([pyarrow] in some cases)
- Int64 (most backends)

[polars]: https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-core/src/datatypes/aliases.rs#L14
[pyarrow]: https://github.com/narwhals-dev/narwhals/blob/bbc5d4492667eb3b9a364caba35e51308c86cf7d/narwhals/_arrow/dataframe.py#L534-L547
"""


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


# NOTE: See (https://github.com/microsoft/pyright/issues/10673#issuecomment-3033789021)
# The issue is `T` possibly being `Iterable`
# Ignoring here still leaks the issue to the caller, where you need to annotate the base case
@overload
def flatten_hash_safe(iterable: Iterable[OneOrIterable[str]], /) -> Iterator[str]: ...
@overload
def flatten_hash_safe(
    iterable: Iterable[OneOrIterable[Series]], /
) -> Iterator[Series]: ...
@overload
def flatten_hash_safe(
    iterable: Iterable[OneOrIterable[CompliantSeries]], /
) -> Iterator[CompliantSeries]: ...
@overload
def flatten_hash_safe(
    iterable: Iterable[OneOrIterable[ColumnNameOrSelector]], /
) -> Iterator[ColumnNameOrSelector]: ...
@overload
def flatten_hash_safe(iterable: Iterable[OneOrIterable[T]], /) -> Iterator[T]: ...
def flatten_hash_safe(iterable: Iterable[OneOrIterable[Any]], /) -> Iterator[Any]:
    """Fully unwrap all levels of nesting.

    Aiming to reduce the chances of passing an unhashable argument.
    """
    for element in iterable:
        if isinstance(element, Iterable) and not is_iterable_reject(element):
            yield from flatten_hash_safe(element)
        else:
            yield element


def _not_one_or_iterable_str_error(obj: Any, /) -> TypeError:
    msg = f"Expected one or an iterable of strings, but got: {qualified_type_name(obj)!r}\n{obj!r}"
    return TypeError(msg)


def ensure_seq_str(obj: OneOrIterable[str], /) -> Seq[str]:
    if not isinstance(obj, Iterable):
        raise _not_one_or_iterable_str_error(obj)
    return (obj,) if isinstance(obj, str) else tuple(obj)


def ensure_list_str(obj: OneOrIterable[str], /) -> list[str]:
    if not isinstance(obj, Iterable):
        raise _not_one_or_iterable_str_error(obj)  # pragma: no cover
    return [obj] if isinstance(obj, str) else list(obj)


def _has_columns(obj: Any) -> TypeIs[_StoresColumns]:
    return _hasattr_static(obj, "columns")


def _reprlib_repr_backport() -> reprlib.Repr:
    # 3.12 added `indent` https://github.com/python/cpython/issues/92734
    # but also a useful constructor https://github.com/python/cpython/issues/94343
    import reprlib

    if sys.version_info >= (3, 12):  # pragma: no cover
        return reprlib.Repr(indent=4, maxlist=10)
    else:  # pragma: no cover  # noqa: RET505
        obj = reprlib.Repr()
        obj.maxlist = 10
        return obj


@overload
def normalize_target_file(target: FileSource) -> str: ...
@overload
def normalize_target_file(target: None) -> None: ...
@overload
def normalize_target_file(target: BytesIO) -> BytesIO: ...
@overload
def normalize_target_file(target: FileSource | BytesIO) -> str | BytesIO: ...
def normalize_target_file(target: FileSource | BytesIO | None) -> str | BytesIO | None:
    if target is None or isinstance(target, (str, BytesIO)):
        return target
    from pathlib import Path

    return str(Path(target))


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

        Notes:
            When an `source` is an `Iterator`, it will only be consumed *iff* the result of
            `temp.column_names` advances at least once.
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
        else:  # pragma: no cover (has coverage, but there's randomness in the test)
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


class todo:  # pragma: no cover  # noqa: N801
    """A variation of `not_implemented`, for shorter-lived placeholders."""

    def __set_name__(self, owner: type[Any], name: str) -> None:
        self._name_owner: str = owner.__name__
        self.__name__: str = name

    def __get__(self, instance: object | None, owner: type[Any] | None, /) -> Any:
        if instance is None:
            return self
        msg = f"TODO: `{self._name_owner}.{self.__name__}(...)`"
        raise NotImplementedError(msg)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        msg = "Stop being clever!"
        raise NotImplementedError(msg)
