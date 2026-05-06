from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, overload

import pyarrow as pa
import pyarrow.compute as pc

from narwhals._plan.arrow.functions._arithmetic import add
from narwhals._plan.arrow.functions._construction import lit

__all__ = ("coalesce", "max_horizontal", "min_horizontal", "reduce", "sum_horizontal")

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from _typeshed import Incomplete

    from narwhals._plan.arrow.typing import (
        ArrowAny,
        BinaryFunction,
        ChunkedOrScalar,
        ChunkedOrScalarAny,
        NumericScalar,
        ScalarPT_contra,
        ScalarRT_co,
    )
    from narwhals._plan.typing import OneOrIterable, Seq
    from narwhals.typing import NonNestedLiteral

    _R_co = TypeVar("_R_co", bound=ChunkedOrScalarAny, covariant=True)

    @overload
    def reduce(
        function: Callable[[ArrowAny, ArrowAny], _R_co],
        iterable: Iterable[ArrowAny],
        initial: Incomplete | None = None,
        /,
    ) -> _R_co: ...
    @overload
    def reduce(
        function: BinaryFunction[ScalarPT_contra, ScalarRT_co],
        iterable: Iterable[ArrowAny],
        initial: Incomplete | None = None,
        /,
    ) -> ChunkedOrScalar[ScalarRT_co]: ...
    def reduce(
        function: Callable[[ArrowAny, ArrowAny], _R_co]
        | BinaryFunction[ScalarPT_contra, ScalarRT_co],
        iterable: Iterable[ArrowAny],
        initial: Incomplete | None = None,
        /,
    ) -> ChunkedOrScalar[ScalarRT_co] | _R_co:
        """Wider overloads to avoid invariant of `functools.reduce`."""
        raise NotImplementedError
else:
    from functools import reduce


# TODO @dangotbanned: Wrap horizontal functions with correct typing
# Should only return scalar if all elements are as well
# NOTE: Changing typing will propagate to a lot of places (so be careful!):
# - `_round.{clip,clip_lower,clip_upper}`
# - `acero.join_asof_tables`
# - `ArrowNamespace.{min,max}_horizontal`
# - `ArrowNamespace.coalesce`
# - `ArrowSeries.rolling_var`
min_horizontal = pc.min_element_wise
max_horizontal = pc.max_element_wise
coalesce = pc.coalesce


def sum_horizontal(
    first: OneOrIterable[ChunkedOrScalarAny], *more: ChunkedOrScalarAny
) -> ChunkedOrScalar[NumericScalar]:
    return reduce(add, _fill_null_flat(first, more, 0))


def _flatten(
    first: OneOrIterable[ChunkedOrScalarAny], more: Seq[ChunkedOrScalarAny]
) -> Iterator[ChunkedOrScalarAny]:
    if isinstance(first, (pa.ChunkedArray, pa.Scalar)):
        yield first
    else:
        yield from first
    yield from more


def _fill_null_flat(
    first: OneOrIterable[ChunkedOrScalarAny],
    more: Seq[ChunkedOrScalarAny],
    fill: NonNestedLiteral = None,
) -> Iterator[ChunkedOrScalarAny]:
    if fill is None:
        yield from _flatten(first, more)
    else:
        yield from (coalesce(native, lit(fill)) for native in _flatten(first, more))
