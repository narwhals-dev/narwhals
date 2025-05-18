from __future__ import annotations

import enum
from typing import TYPE_CHECKING

from narwhals._plan.common import Immutable

if TYPE_CHECKING:
    from narwhals._plan.common import Seq
    from narwhals.typing import RankMethod


class FunctionFlags(enum.Flag):
    ALLOW_GROUP_AWARE = 1 << 0
    """> Raise if use in group by

    Not sure where this is disabled.
    """

    INPUT_WILDCARD_EXPANSION = 1 << 4
    """Appears on all the horizontal aggs.

    https://github.com/pola-rs/polars/blob/e8ad1059721410e65a3d5c1d84055fb22a4d6d43/crates/polars-plan/src/plans/options.rs#L49-L58
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
    """ExprMetadata` but less god object.

    https://github.com/pola-rs/polars/blob/3fd7ecc5f9de95f62b70ea718e7e5dbf951b6d1c/crates/polars-plan/src/plans/options.rs
    """

    __slots__ = ("flags",)

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


# TODO @dangotbanned: Decide on constructors
class SortOptions(Immutable):
    __slots__ = ("descending", "nulls_last")

    descending: bool
    nulls_last: bool

    def __repr__(self) -> str:
        args = f"descending={self.descending!r}, nulls_last={self.nulls_last!r}"
        return f"{type(self).__name__}({args})"


class SortMultipleOptions(Immutable):
    __slots__ = ("descending", "nulls_last")

    descending: Seq[bool]
    nulls_last: Seq[bool]

    def __repr__(self) -> str:
        args = (
            f"descending={list(self.descending)!r}, nulls_last={list(self.nulls_last)!r}"
        )
        return f"{type(self).__name__}({args})"


class RankOptions(Immutable):
    """https://github.com/narwhals-dev/narwhals/pull/2555."""

    __slots__ = ("descending", "method")

    method: RankMethod
    descending: bool


class EWMOptions(Immutable):
    """Deviates from polars, since we aren't pre-computing alpha.

    https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-arrow/src/legacy/kernels/ewm/mod.rs#L14-L20
    """

    __slots__ = (
        "adjust",
        "alpha",
        "com",
        "half_life",
        "ignore_nulls",
        "min_samples",
        "span",
    )

    com: float | None
    span: float | None
    half_life: float | None
    alpha: float | None
    adjust: bool
    min_samples: int
    ignore_nulls: bool


class RollingVarParams(Immutable):
    __slots__ = ("ddof",)

    ddof: int


class RollingOptionsFixedWindow(Immutable):
    """https://github.com/pola-rs/polars/blob/dafd0a2d0e32b52bcfa4273bffdd6071a0d5977a/crates/polars-core/src/chunked_array/ops/rolling_window.rs#L10-L23."""

    __slots__ = ("center", "fn_params", "min_samples", "window_size")

    window_size: int
    min_samples: int
    """Renamed from `min_periods`, re-uses `window_size` if null."""

    center: bool
    fn_params: RollingVarParams | None
