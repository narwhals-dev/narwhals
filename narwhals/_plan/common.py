from __future__ import annotations

import sys
from collections.abc import Iterable
from copy import deepcopy
from io import BytesIO
from secrets import token_hex
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast, overload

from narwhals._plan._guards import is_iterable_reject
from narwhals._utils import _hasattr_static, qualified_type_name
from narwhals.dtypes import DType
from narwhals.exceptions import NarwhalsError

if TYPE_CHECKING:
    import reprlib
    from collections.abc import Iterator
    from typing import Any, ClassVar, TypeVar

    from typing_extensions import TypeAlias, TypeIs

    from narwhals._plan.compliant.series import CompliantSeries
    from narwhals._plan.series import Series
    from narwhals._plan.typing import (
        ClosedKwds,
        ColumnNameOrSelector,
        DTypeT,
        NonNestedDTypeT,
        OneOrIterable,
        Seq,
    )
    from narwhals._utils import _StoresColumns
    from narwhals.typing import FileSource

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

    __slots__ = ("__name__", "_name_owner")

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


_Proxy: TypeAlias = "MappingProxyType[str, Any]"


def closed_kwds(
    strategy: Literal["reusable", "single_copy", "zero_copy"] = "zero_copy", **kwds: Any
) -> ClosedKwds:
    """Create a callable that returns *closed-over* keyword-only arguments.

    Arguments:
        strategy: {"reusable", "single_copy", "zero_copy"}

            Strategy to balance mutability and memory usage.

            - *"reusable"*: Updates to mutable *results* persist only for the result of the **current** call.
            - *"single_copy"*: Updates to mutable *results* will be reflected across **all** calls.
            - *"zero_copy"*: Updates to mutable *arguments* or the *results* will be reflected across **all** calls.
        **kwds: Arbitrary keyword-only arguments.

    Examples:
        By default, this provides a lightweight wrapper for unhashable objects:

        >>> f = closed_kwds(a={"mutable": [1, 2, 3]}, b=[4, 5, 6])
        >>> hash(f) == hash(f)
        True

        >>> f == f
        True

        >>> f()
        mappingproxy({'a': {'mutable': [1, 2, 3]}, 'b': [4, 5, 6]})

        `strategy` comparison:

        >>> def show_mutation(
        ...     strategy: Literal["reusable", "single_copy", "zero_copy"],
        ...     mutable: list[int],
        ... ) -> None:
        ...     function = closed_kwds(strategy, a="dog", b=mutable)
        ...     result = function()
        ...     print("First result    :", result)
        ...     mutable.append(2)
        ...     print("Mutated argument:", result)
        ...     result["b"].append(3)
        ...     print("Mutated result  :", result)
        ...     print("Second result   :", function())

        `"zero_copy"` may be reasonable if any of these apply:
        - all arguments are immutable
        - their types do not support `copy.deepcopy`
        - making a single copy would be prohibitive

        >>> show_mutation("zero_copy", [1])
        First result    : {'a': 'dog', 'b': [1]}
        Mutated argument: {'a': 'dog', 'b': [1, 2]}
        Mutated result  : {'a': 'dog', 'b': [1, 2, 3]}
        Second result   : {'a': 'dog', 'b': [1, 2, 3]}

        `"single_copy"` provides some extra safety and should be considered if:
        - at least one argument may be mutable
        - the returned function will only be called once
        - the result of the function will not be mutated

        >>> show_mutation("single_copy", [1])
        First result    : {'a': 'dog', 'b': [1]}
        Mutated argument: {'a': 'dog', 'b': [1]}
        Mutated result  : {'a': 'dog', 'b': [1, 3]}
        Second result   : {'a': 'dog', 'b': [1, 3]}

        `"reusable"` is both the safest and most expensive strategy, consider if:
        - both mutable arguments and updates to them are expected
        - the returned function will be called multiple times, and each should
            be isolated from previous calls
        - the cost of `copy.deepcopy`(s) is small/irrelevant

        >>> show_mutation("reusable", [1])
        First result    : {'a': 'dog', 'b': [1]}
        Mutated argument: {'a': 'dog', 'b': [1]}
        Mutated result  : {'a': 'dog', 'b': [1, 3]}
        Second result   : {'a': 'dog', 'b': [1]}
    """
    if not kwds:
        return _closed_kwds_empty
    if strategy == "zero_copy":
        return _closed_kwds_impl(kwds, copy=False)
    if strategy == "single_copy":
        return _closed_kwds_impl(kwds, copy=True)
    return _closed_kwds_reusable_impl(kwds)


_EMPTY_MAPPING_PROXY: _Proxy = MappingProxyType({})


def _closed_kwds_empty() -> _Proxy:
    return _EMPTY_MAPPING_PROXY


def _closed_kwds_impl(mapping: dict[str, Any] | _Proxy, /, *, copy: bool) -> ClosedKwds:
    # NOTE: `proxy` is assigned like this so `_.__closure__` has 1 cell and is not a `dict`
    proxy = MappingProxyType(
        mapping
        if not copy
        else deepcopy(mapping if isinstance(mapping, dict) else mapping.copy())
    )

    def _() -> _Proxy:
        return proxy

    return _


def _closed_kwds_reusable_impl(mapping: dict[str, Any], /) -> ClosedKwds:
    # ensure the next call produces the original `mapping`, while removing the input from this scope
    next_copy = _closed_kwds_impl(mapping, copy=True)
    del mapping

    def iter_clone() -> Iterator[_Proxy]:
        # infinte generator, maintaining the `next_copy` and exposing already deepcopy'd output
        nonlocal next_copy
        while True:
            out = next_copy()
            next_copy = _closed_kwds_impl(out, copy=True)
            yield out

    clones = iter(iter_clone())

    def _() -> _Proxy:
        return next(clones)

    return _
