"""`ExprMetadata` but less god object.

- https://github.com/pola-rs/polars/blob/3fd7ecc5f9de95f62b70ea718e7e5dbf951b6d1c/crates/polars-plan/src/plans/options.rs
"""

from __future__ import annotations

import enum

from narwhals._plan.common import Immutable


class FunctionFlags(enum.Flag):
    ALLOW_GROUP_AWARE = 1 << 0
    """> Raise if use in group by

    Not sure where this is disabled.
    """

    RETURNS_SCALAR = 1 << 5
    """Automatically explode on unit length if it ran as final aggregation."""

    ROW_SEPARABLE = 1 << 8
    """Not sure lol.

    https://github.com/pola-rs/polars/pull/22573
    """

    LENGTH_PRESERVING = 1 << 9
    """mutually exclusive with `RETURNS_SCALAR`"""

    def is_elementwise(self) -> bool:
        return self in (FunctionFlags.ROW_SEPARABLE | FunctionFlags.LENGTH_PRESERVING)

    def returns_scalar(self) -> bool:
        return self in FunctionFlags.RETURNS_SCALAR

    def is_length_preserving(self) -> bool:
        return self in FunctionFlags.LENGTH_PRESERVING

    @staticmethod
    def default() -> FunctionFlags:
        return FunctionFlags.ALLOW_GROUP_AWARE


class FunctionOptions(Immutable):
    flags: FunctionFlags

    def is_elementwise(self) -> bool:
        return self.flags.is_elementwise()

    def returns_scalar(self) -> bool:
        return self.flags.returns_scalar()

    def is_length_preserving(self) -> bool:
        return self.flags.is_length_preserving()

    def with_flags(self, flags: FunctionFlags, /) -> FunctionOptions:
        if (FunctionFlags.RETURNS_SCALAR | FunctionFlags.LENGTH_PRESERVING) in flags:
            msg = "A function cannot both return a scalar and preserve length, they are mutually exclusive."
            raise TypeError(msg)
        obj = FunctionOptions.__new__(FunctionOptions)
        object.__setattr__(obj, "flags", self.flags | flags)
        return obj

    def with_elementwise(self) -> FunctionOptions:
        return self.with_flags(
            FunctionFlags.ROW_SEPARABLE | FunctionFlags.LENGTH_PRESERVING
        )

    @staticmethod
    def default() -> FunctionOptions:
        obj = FunctionOptions.__new__(FunctionOptions)
        object.__setattr__(obj, "flags", FunctionFlags.default())
        return obj

    @staticmethod
    def elementwise() -> FunctionOptions:
        return FunctionOptions.default().with_elementwise()

    @staticmethod
    def row_separable() -> FunctionOptions:
        return FunctionOptions.groupwise().with_flags(FunctionFlags.ROW_SEPARABLE)

    @staticmethod
    def length_preserving() -> FunctionOptions:
        return FunctionOptions.default().with_flags(FunctionFlags.LENGTH_PRESERVING)

    @staticmethod
    def groupwise() -> FunctionOptions:
        return FunctionOptions.default()

    @staticmethod
    def aggregation() -> FunctionOptions:
        return FunctionOptions.groupwise().with_flags(FunctionFlags.RETURNS_SCALAR)
