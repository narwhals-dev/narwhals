from __future__ import annotations

from typing import TYPE_CHECKING, Final, overload

import pyarrow as pa

from narwhals._plan._guards import NON_NESTED_LITERAL_TPS
from narwhals._plan.arrow.functions._arithmetic import add
from narwhals._plan.arrow.functions._bin_op import and_, or_
from narwhals._plan.arrow.functions.meta import call

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from narwhals._plan.arrow.typing import (
        BinaryFunction,
        BooleanScalar,
        ChunkedArrayAny,
        ChunkedOrScalar,
        IntoArrowAny,
        IntoNative,
        Native,
        NumericScalar,
        ScalarAny,
        ScalarPT_contra as ScalarP,
        ScalarRT_co,
    )
    from narwhals._plan.typing import OneOrIterable, Seq

    def reduce(
        function: BinaryFunction[ScalarP, ScalarRT_co],
        iterable: Iterable[IntoArrowAny],
        initial: IntoArrowAny = None,
        /,
    ) -> ChunkedOrScalar[ScalarRT_co]:
        """`functools.reduce` with typing that's friendlier to `pyarrow`.

        The version in [`typeshed`] isn't compatible with broadcasting + type promotion.

        [`typeshed`]: https://github.com/python/typeshed/blob/d541d5a20cb7533c6eaba1430b5004f9b0e504ec/stdlib/functools.pyi#L34-L43
        """
        raise NotImplementedError
else:
    from functools import reduce


__all__ = (
    "all_horizontal",
    "any_horizontal",
    "coalesce",
    "max_horizontal",
    "min_horizontal",
    "reduce",
    "sum_horizontal",
)


def _coalesce(*args: IntoNative) -> Native:
    result: Native = call("coalesce", *args)
    return result


# TODO @dangotbanned: These can still be shrunk down some more
def any_horizontal(
    first: OneOrIterable[Native], *more: Native, ignore_nulls: bool = False
) -> ChunkedOrScalar[BooleanScalar]:
    flat = _flatten(first, more)
    return reduce(or_, (_coalesce(arr, False) for arr in flat) if ignore_nulls else flat)


def all_horizontal(
    first: OneOrIterable[Native], *more: Native, ignore_nulls: bool = False
) -> ChunkedOrScalar[BooleanScalar]:
    flat = _flatten(first, more)
    return reduce(and_, (_coalesce(arr, True) for arr in flat) if ignore_nulls else flat)


def sum_horizontal(
    first: OneOrIterable[IntoNative], *more: IntoNative
) -> ChunkedOrScalar[NumericScalar]:
    return reduce(add, (_coalesce(arr, 0) for arr in _flatten(first, more)))


def coalesce(first: OneOrIterable[IntoNative], *more: IntoNative) -> Native:
    return _coalesce(*_flatten(first, more))


@overload
def min_horizontal(first: ScalarAny, *more: ScalarAny) -> ScalarAny: ...
@overload
def min_horizontal(first: ChunkedArrayAny, *more: IntoNative) -> ChunkedArrayAny: ...
@overload
def min_horizontal(first: OneOrIterable[IntoNative], *more: IntoNative) -> Native: ...
def min_horizontal(first: OneOrIterable[IntoNative], *more: IntoNative) -> Native:
    result: Native = call("min_element_wise", *_flatten(first, more))
    return result


@overload
def max_horizontal(first: ScalarAny, *more: ScalarAny) -> ScalarAny: ...
@overload
def max_horizontal(first: ChunkedArrayAny, *more: IntoNative) -> ChunkedArrayAny: ...
@overload
def max_horizontal(first: OneOrIterable[IntoNative], *more: IntoNative) -> Native: ...
def max_horizontal(first: OneOrIterable[IntoNative], *more: IntoNative) -> Native:
    result: Native = call("max_element_wise", *_flatten(first, more))
    return result


_NON_ITERABLE: Final = (pa.ChunkedArray, pa.Scalar, *NON_NESTED_LITERAL_TPS, type(None))


def _flatten(
    first: OneOrIterable[IntoNative], more: Seq[IntoNative]
) -> Iterator[IntoNative]:
    if isinstance(first, _NON_ITERABLE):
        yield first
    else:
        yield from first
    yield from more
