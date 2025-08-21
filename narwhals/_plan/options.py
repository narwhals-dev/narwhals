from __future__ import annotations

import enum
from itertools import repeat
from typing import TYPE_CHECKING, Literal

from narwhals._plan.common import Immutable

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import pyarrow.compute as pc

    from narwhals._plan.typing import Seq
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
    """Given a function `f` and a column of values `[v1, ..., vn]`.

    `f` is row-separable *iff*:

        f([v1, ..., vn]) = concat(f(v1, ... vm), f(vm+1, ..., vn))

    In isolation, used on `drop_nulls`, `int_range`

    https://github.com/pola-rs/polars/pull/22573
    """

    LENGTH_PRESERVING = 1 << 9
    """mutually exclusive with `RETURNS_SCALAR`"""

    def is_elementwise(self) -> bool:
        return (FunctionFlags.ROW_SEPARABLE | FunctionFlags.LENGTH_PRESERVING) in self

    def returns_scalar(self) -> bool:
        return FunctionFlags.RETURNS_SCALAR in self

    def is_length_preserving(self) -> bool:
        return FunctionFlags.LENGTH_PRESERVING in self

    def is_row_separable(self) -> bool:
        return FunctionFlags.ROW_SEPARABLE in self

    def is_input_wildcard_expansion(self) -> bool:
        return FunctionFlags.INPUT_WILDCARD_EXPANSION in self

    @staticmethod
    def default() -> FunctionFlags:
        return FunctionFlags.ALLOW_GROUP_AWARE

    def __str__(self) -> str:
        name = self.name or "<FUNCTION_FLAGS_UNKNOWN>"
        return name.replace("|", " | ")


class FunctionOptions(Immutable):
    """ExprMetadata` but less god object.

    https://github.com/pola-rs/polars/blob/3fd7ecc5f9de95f62b70ea718e7e5dbf951b6d1c/crates/polars-plan/src/plans/options.rs
    """

    __slots__ = ("flags",)
    flags: FunctionFlags

    def __str__(self) -> str:
        return f"{type(self).__name__}(flags='{self.flags}')"

    def is_elementwise(self) -> bool:
        return self.flags.is_elementwise()

    def returns_scalar(self) -> bool:
        return self.flags.returns_scalar()

    def is_length_preserving(self) -> bool:
        return self.flags.is_length_preserving()

    def is_row_separable(self) -> bool:
        return self.flags.is_row_separable()

    def is_input_wildcard_expansion(self) -> bool:
        return self.flags.is_input_wildcard_expansion()

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

    @staticmethod
    def horizontal() -> FunctionOptions:
        return FunctionOptions.elementwise().with_flags(
            FunctionFlags.INPUT_WILDCARD_EXPANSION
        )


class SortOptions(Immutable):
    __slots__ = ("descending", "nulls_last")
    descending: bool
    nulls_last: bool

    def __repr__(self) -> str:
        args = f"descending={self.descending!r}, nulls_last={self.nulls_last!r}"
        return f"{type(self).__name__}({args})"

    @staticmethod
    def default() -> SortOptions:
        return SortOptions(descending=False, nulls_last=False)

    def to_arrow(self) -> pc.ArraySortOptions:
        import pyarrow.compute as pc

        return pc.ArraySortOptions(
            order=("descending" if self.descending else "ascending"),
            null_placement=("at_end" if self.nulls_last else "at_start"),
        )

    def to_multiple(self, n_repeat: int = 1, /) -> SortMultipleOptions:
        if n_repeat == 1:
            desc: Seq[bool] = (self.descending,)
            nulls: Seq[bool] = (self.nulls_last,)
        else:
            desc = tuple(repeat(self.descending, n_repeat))
            nulls = tuple(repeat(self.nulls_last))
        return SortMultipleOptions(descending=desc, nulls_last=nulls)


class SortMultipleOptions(Immutable):
    __slots__ = ("descending", "nulls_last")
    descending: Seq[bool]
    nulls_last: Seq[bool]

    def __repr__(self) -> str:
        args = (
            f"descending={list(self.descending)!r}, nulls_last={list(self.nulls_last)!r}"
        )
        return f"{type(self).__name__}({args})"

    @staticmethod
    def parse(
        *, descending: bool | Iterable[bool], nulls_last: bool | Iterable[bool]
    ) -> SortMultipleOptions:
        desc = (descending,) if isinstance(descending, bool) else tuple(descending)
        nulls = (nulls_last,) if isinstance(nulls_last, bool) else tuple(nulls_last)
        return SortMultipleOptions(descending=desc, nulls_last=nulls)

    def to_arrow(self, by: Sequence[str]) -> pc.SortOptions:
        import pyarrow.compute as pc

        first = self.nulls_last[0]
        if len(self.nulls_last) != 1 and any(x != first for x in self.nulls_last[1:]):
            msg = f"pyarrow doesn't support multiple values for `nulls_last`, got: {self.nulls_last!r}"
            raise NotImplementedError(msg)
        if len(self.descending) == 1:
            descending: Iterable[bool] = repeat(self.descending[0], len(by))
        else:
            descending = self.descending
        sorting: list[tuple[str, Literal["ascending", "descending"]]] = [
            (key, "descending" if desc else "ascending")
            for key, desc in zip(by, descending)
        ]
        placement: Literal["at_start", "at_end"] = "at_end" if first else "at_start"
        return pc.SortOptions(sort_keys=sorting, null_placement=placement)


class RankOptions(Immutable):
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
    """Renamed from `min_periods`, reuses `window_size` if null."""
    center: bool
    fn_params: RollingVarParams | None
