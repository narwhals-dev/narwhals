from __future__ import annotations

from collections.abc import Iterable, Sized
from typing import TYPE_CHECKING, Protocol

from narwhals._plan.compliant.typing import (
    FrameT_contra,
    HasVersion,
    LengthT,
    NamespaceT_co,
    R_co,
    SeriesT,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from typing_extensions import Self

    from narwhals._plan import expressions as ir


class SupportsBroadcast(Protocol[SeriesT, LengthT]):
    """Minimal broadcasting for `Expr` results."""

    def _length(self) -> LengthT:
        """Return the length of the current expression."""
        ...

    @classmethod
    def _length_all(
        cls, exprs: Sequence[SupportsBroadcast[SeriesT, LengthT]], /
    ) -> Sequence[LengthT]:
        return [e._length() for e in exprs]

    @classmethod
    def _length_max(cls, lengths: Sequence[LengthT], /) -> LengthT:
        """Return the maximum length among `exprs`."""
        ...

    @classmethod
    def _length_required(
        cls,
        exprs: Sequence[SupportsBroadcast[SeriesT, LengthT]],
        /,
        default: LengthT | None = None,
    ) -> LengthT | None:
        """Return the broadcast length, if all lengths do not equal the maximum."""

    @classmethod
    def align(
        cls,
        exprs: Iterable[SupportsBroadcast[SeriesT, LengthT]],
        /,
        default: LengthT | None = None,
    ) -> Iterator[SeriesT]:
        """Yield broadcasted `Scalar`s and unwrapped `Expr`s from `exprs`.

        `default` must be provided when operating in a `with_columns` context.
        """
        exprs = tuple(exprs)
        length = default if len(exprs) == 1 else cls._length_required(exprs, default)
        if length is None:
            for e in exprs:
                yield e.to_series()
        else:
            for e in exprs:
                yield e.broadcast(length)

    def broadcast(self, length: LengthT, /) -> SeriesT:
        """Repeat a `Scalar`, or unwrap an `Expr` into a `Series`.

        For `Scalar`, this is always safe, but will be less efficient than if we can operate on (`Scalar`, `Series`).

        For `Expr`, mismatched `length` will raise, but the operation is otherwise free.
        """
        ...

    @classmethod
    def from_series(cls, series: SeriesT, /) -> Self: ...
    def to_series(self) -> SeriesT: ...


class EagerBroadcast(Sized, SupportsBroadcast[SeriesT, int], Protocol[SeriesT]):
    """Determines expression length via the size of the container."""

    def _length(self) -> int:
        return len(self)

    @classmethod
    def _length_max(cls, lengths: Sequence[int], /) -> int:
        return max(lengths)

    @classmethod
    def _length_required(
        cls,
        exprs: Sequence[SupportsBroadcast[SeriesT, int]],
        /,
        default: int | None = None,
    ) -> int | None:
        lengths = cls._length_all(exprs)
        max_length = cls._length_max(lengths)
        required = any(len_ != max_length for len_ in lengths)
        return max_length if required else default


class ExprDispatch(HasVersion, Protocol[FrameT_contra, R_co, NamespaceT_co]):
    # NOTE: Needs to stay `covariant` and never be used as a parameter
    def __narwhals_namespace__(self) -> NamespaceT_co: ...
    @classmethod
    def from_ir(cls, node: ir.ExprIR, frame: FrameT_contra, name: str) -> R_co:
        obj = cls.__new__(cls)
        obj._version = frame.version
        return node.dispatch(obj, frame, name)

    @classmethod
    def from_named_ir(cls, named_ir: ir.NamedIR[ir.ExprIR], frame: FrameT_contra) -> R_co:
        return cls.from_ir(named_ir.expr, frame, named_ir.name)
