"""Create known-length `pa.Array`s, filled with a single value.

Note:
    These wrappers should be preferred when the lack of precision in input types causes
    false negatives and/or LSP hangs in the `pyarrow-stubs` overloads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa  # ignore-banned-import

from narwhals._plan.arrow.functions._construction import lit

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals._plan.arrow.typing import ArrayAny, ArrowAny, ScalarAny
    from narwhals.typing import NonNestedLiteral

Incomplete: TypeAlias = Any

__all__ = ["nulls_like", "repeat", "repeat_like", "repeat_unchecked", "zeros"]


def nulls_like(n: int, /, native: ArrowAny) -> ArrayAny:
    """Create an Array of length `n` filled with nulls.

    Uses the type of `native`, where `pa.nulls` defaults to `pa.NullType`.
    """
    result: ArrayAny = pa.nulls(n, native.type)
    return result


def repeat(value: ScalarAny | NonNestedLiteral, /, n: int) -> ArrayAny:
    """Create an Array of length `n` filled with the given `value`.

    Adds an additional check and coerces `NonNestedLiteral` through `pa.Scalar`.

    Tip:
        If you *already* know `pa.Scalar` is the only possible input,
        use `repeat_unchecked` instead.
    """
    value = value if isinstance(value, pa.Scalar) else lit(value)
    return repeat_unchecked(value, n)


def repeat_like(value: NonNestedLiteral, /, n: int, native: ArrowAny) -> ArrayAny:
    """Create an Array of length `n` filled with the given `value`.

    Uses the type of `native`.
    """
    return repeat_unchecked(lit(value, native.type), n)


def repeat_unchecked(value: ScalarAny, /, n: int) -> ArrayAny:
    """Create an Array of length `n` filled with the given `value`."""
    repeat_: Incomplete = pa.repeat
    result: ArrayAny = repeat_(value, n)
    return result


def zeros(n: int, /) -> pa.Int64Array:
    """Create an Array of length `n` filled with zeros."""
    return pa.repeat(0, n)
