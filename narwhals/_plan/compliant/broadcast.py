from __future__ import annotations

from collections.abc import Collection, Iterable, Sized
from typing import TYPE_CHECKING, Protocol

from narwhals._plan.compliant.typing import NativeSeriesT, NativeSeriesT_co

if TYPE_CHECKING:
    from collections.abc import Iterator

    from narwhals._plan.compliant import CompliantSeries as Series

__all__ = ["BroadcastFrom", "BroadcastSeries"]


class BroadcastSeries(Protocol[NativeSeriesT]):
    """Mostly implemented mixin protocol for eager expression broadcasting.

    `[NativeSeriesT]`.

    The missing parts are split out into `BroadcastFrom`, to resolve a variance conundrum.
    """

    __slots__ = ()

    @classmethod
    def align(
        cls, exprs: Iterable[BroadcastFrom[NativeSeriesT]], /, default: int | None = None
    ) -> Iterator[Series[NativeSeriesT]]:
        """Yield broadcasted `Scalar`s and unwrapped `Expr`s from `exprs`.

        `default` must be provided when operating in a `with_columns` context.
        """
        exprs = tuple(exprs)
        length = (
            default
            if len(exprs) == 1
            else cls._length_required([len(e) for e in exprs], default)
        )
        if length is None:
            yield from (e.to_series() for e in exprs)
        else:
            yield from (e.broadcast(length) for e in exprs)

    @staticmethod
    def _length_required(
        lengths: Collection[int], /, default: int | None = None
    ) -> int | None:
        """Return the broadcast length, if all lengths do not equal the maximum."""
        max_length = max(lengths)
        required = any(len_ != max_length for len_ in lengths)
        return max_length if required else default


class BroadcastFrom(Sized, Protocol[NativeSeriesT_co]):
    """Requirements for an eager expression representation to be used in `BroadcastSeries`.

    `[NativeSeriesT_co]`.
    """

    __slots__ = ()

    def to_series(self) -> Series[NativeSeriesT_co]: ...
    def broadcast(self, length: int, /) -> Series[NativeSeriesT_co]:
        """Repeat a `Scalar`, or unwrap an `Expr` into a `Series`.

        For `Scalar`, this is always safe, but will be less efficient than if we can operate on (`Scalar`, `Series`).

        For `Expr`, mismatched `length` will raise, but the operation is otherwise free.
        """
        ...
