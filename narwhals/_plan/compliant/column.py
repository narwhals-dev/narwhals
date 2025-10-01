from __future__ import annotations

from collections.abc import Sequence, Sized
from typing import TYPE_CHECKING, Protocol

from narwhals._plan.common import flatten_hash_safe
from narwhals._plan.compliant.typing import LengthT, SeriesT

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from typing_extensions import Self

    from narwhals._plan.typing import OneOrIterable


class SupportsBroadcast(Protocol[SeriesT, LengthT]):
    """Minimal broadcasting for `Expr` results."""

    @classmethod
    def from_series(cls, series: SeriesT, /) -> Self: ...
    def to_series(self) -> SeriesT: ...
    def broadcast(self, length: LengthT, /) -> SeriesT: ...
    def _length(self) -> LengthT:
        """Return the length of the current expression."""
        ...

    @classmethod
    def _length_max(cls, lengths: Sequence[LengthT], /) -> LengthT:
        """Return the maximum length among `exprs`."""
        ...

    @classmethod
    def _length_required(
        cls, exprs: Sequence[SupportsBroadcast[SeriesT, LengthT]], /
    ) -> LengthT | None:
        """Return the broadcast length, if all lengths do not equal the maximum."""

    @classmethod
    def _length_all(
        cls, exprs: Sequence[SupportsBroadcast[SeriesT, LengthT]], /
    ) -> Sequence[LengthT]:
        return [e._length() for e in exprs]

    @classmethod
    def align(
        cls, *exprs: OneOrIterable[SupportsBroadcast[SeriesT, LengthT]]
    ) -> Iterator[SeriesT]:
        exprs = tuple[SupportsBroadcast[SeriesT, LengthT], ...](flatten_hash_safe(exprs))
        length = cls._length_required(exprs)
        if length is None:
            for e in exprs:
                yield e.to_series()
        else:
            for e in exprs:
                yield e.broadcast(length)


class EagerBroadcast(Sized, SupportsBroadcast[SeriesT, int], Protocol[SeriesT]):
    """Determines expression length via the size of the container."""

    def _length(self) -> int:
        return len(self)

    @classmethod
    def _length_max(cls, lengths: Sequence[int], /) -> int:
        return max(lengths)

    @classmethod
    def _length_required(
        cls, exprs: Sequence[SupportsBroadcast[SeriesT, int]], /
    ) -> int | None:
        lengths = cls._length_all(exprs)
        max_length = cls._length_max(lengths)
        required = any(len_ != max_length for len_ in lengths)
        return max_length if required else None
