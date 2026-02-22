from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Generic, Literal, get_args

from narwhals._plan._immutable import Immutable
from narwhals._plan.common import ensure_seq_str
from narwhals._typing_compat import TypeVar
from narwhals._utils import Implementation, ensure_type
from narwhals.exceptions import InvalidOperationError
from narwhals.typing import AsofJoinStrategy, JoinStrategy, UniqueKeepStrategy

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyarrow.compute as pc
    from typing_extensions import Self

    from narwhals._plan.typing import Accessor, NonCrossJoinStrategy, OneOrIterable, Seq
    from narwhals._typing import Backend
    from narwhals.typing import AsofJoinStrategy as JoinAsofStrategy, RankMethod


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
    """In isolation, means that the function is dependent on the context of surrounding rows.

    Mutually exclusive with `RETURNS_SCALAR`.
    """

    def is_elementwise(self) -> bool:
        return (FunctionFlags.ROW_SEPARABLE | FunctionFlags.LENGTH_PRESERVING) in self

    def returns_scalar(self) -> bool:
        return FunctionFlags.RETURNS_SCALAR in self

    def is_length_preserving(self) -> bool:
        return FunctionFlags.LENGTH_PRESERVING in self  # pragma: no cover

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


_INVALID = FunctionFlags.RETURNS_SCALAR | FunctionFlags.LENGTH_PRESERVING


class FunctionOptions(Immutable):
    """https://github.com/pola-rs/polars/blob/3fd7ecc5f9de95f62b70ea718e7e5dbf951b6d1c/crates/polars-plan/src/plans/options.rs"""  # noqa: D415

    __slots__ = ("flags",)
    flags: FunctionFlags

    def __str__(self) -> str:
        return f"{type(self).__name__}(flags='{self.flags}')"

    def is_elementwise(self) -> bool:
        return self.flags.is_elementwise()

    def returns_scalar(self) -> bool:
        return self.flags.returns_scalar()

    def is_length_preserving(self) -> bool:
        return self.flags.is_length_preserving()  # pragma: no cover

    def is_row_separable(self) -> bool:
        return self.flags.is_row_separable()

    def is_input_wildcard_expansion(self) -> bool:
        return self.flags.is_input_wildcard_expansion()

    def with_flags(self, flags: FunctionFlags, /) -> FunctionOptions:
        new_flags = self.flags | flags
        if _INVALID in new_flags:
            msg = "A function cannot both return a scalar and preserve length, they are mutually exclusive."
            raise TypeError(msg)
        obj = FunctionOptions.__new__(FunctionOptions)
        object.__setattr__(obj, "flags", new_flags)
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

    def to_arrow(self, by: Sequence[str]) -> pc.SortOptions:
        from narwhals._plan.arrow.options import sort

        return sort(*by, descending=self.descending, nulls_last=self.nulls_last)

    def _ensure_single_nulls_last(self, backend: Backend, /) -> bool:
        return self.nulls_last


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
    def default() -> SortMultipleOptions:  # pragma: no cover
        return SortMultipleOptions(descending=(False,), nulls_last=(False,))

    @staticmethod
    def parse(
        *, descending: OneOrIterable[bool], nulls_last: OneOrIterable[bool]
    ) -> SortMultipleOptions:
        desc = (descending,) if isinstance(descending, bool) else tuple(descending)
        nulls = (nulls_last,) if isinstance(nulls_last, bool) else tuple(nulls_last)
        return SortMultipleOptions(descending=desc, nulls_last=nulls)

    def _ensure_single_nulls_last(self, backend: Backend, /) -> bool:
        first = self.nulls_last[0]
        if len(self.nulls_last) != 1 and any(x != first for x in self.nulls_last[1:]):
            msg = f"{Implementation.from_backend(backend)!r} does not support multiple values for `nulls_last`, got: {self.nulls_last!r}"  # pragma: no cover
            raise NotImplementedError(msg)
        return first

    def to_arrow(self, by: Sequence[str]) -> pc.SortOptions:
        from narwhals._plan.arrow.options import sort

        nulls_last = self._ensure_single_nulls_last("pyarrow")
        return sort(*by, descending=self.descending, nulls_last=nulls_last)


class RankOptions(Immutable):
    __slots__ = ("descending", "method")
    method: RankMethod
    descending: bool

    def to_arrow(self) -> pc.RankOptions:
        if self.method == "average":  # pragma: no cover
            msg = f"`RankOptions.to_arrow` is not compatible with {self.method=}."
            raise InvalidOperationError(msg)
        from narwhals._plan.arrow.options import rank

        return rank(self.method, descending=self.descending)


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
    center: bool
    fn_params: RollingVarParams | None

    @property
    def ddof(self) -> int:
        return 1 if self.fn_params is None else self.fn_params.ddof


def rolling_options(
    window_size: int, min_samples: int | None, /, *, center: bool, ddof: int | None = None
) -> RollingOptionsFixedWindow:
    ensure_type(window_size, int, param_name="window_size")
    ensure_type(min_samples, int, type(None), param_name="min_samples")
    if window_size < 1:
        msg = "`window_size` must be >= 1"
        raise InvalidOperationError(msg)
    if min_samples is None:
        min_samples = window_size
    elif min_samples < 1:
        msg = "`min_samples` must be >= 1"
        raise InvalidOperationError(msg)
    elif min_samples > window_size:
        msg = "`min_samples` must be <= `window_size`"
        raise InvalidOperationError(msg)
    return RollingOptionsFixedWindow(
        window_size=window_size,
        min_samples=min_samples,
        center=center,
        fn_params=ddof if ddof is None else RollingVarParams(ddof=ddof),
    )


class _BaseIROptions(Immutable):
    __slots__ = ("is_namespaced", "override_name")
    is_namespaced: bool
    override_name: str

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def default(cls) -> Self:  # pragma: no cover[abstract]
        return cls(is_namespaced=False, override_name="")

    @classmethod
    def renamed(cls, name: str, /) -> Self:
        from narwhals._plan.common import replace

        return replace(cls.default(), override_name=name)

    @classmethod
    def namespaced(cls, override_name: str = "", /) -> Self:
        from narwhals._plan.common import replace

        return replace(cls.default(), is_namespaced=True, override_name=override_name)


class ExprIROptions(_BaseIROptions):
    __slots__ = ("allow_dispatch",)
    allow_dispatch: bool

    @classmethod
    def default(cls) -> Self:
        return cls(is_namespaced=False, override_name="", allow_dispatch=True)

    @staticmethod
    def no_dispatch() -> ExprIROptions:
        return ExprIROptions(is_namespaced=False, override_name="", allow_dispatch=False)


class FunctionExprOptions(_BaseIROptions):
    __slots__ = ("accessor_name",)
    accessor_name: Accessor | None
    """Namespace accessor name, if any."""

    @classmethod
    def default(cls) -> Self:
        return cls(is_namespaced=False, override_name="", accessor_name=None)


FEOptions = FunctionExprOptions


class ExplodeOptions(Immutable):
    __slots__ = ("empty_as_null", "keep_nulls")
    empty_as_null: bool
    """Explode an empty list into a `null`."""
    keep_nulls: bool
    """Explode a `null` into a `null`."""

    @staticmethod
    def default() -> ExplodeOptions:  # pragma: no cover
        return ExplodeOptions(empty_as_null=True, keep_nulls=True)

    def any(self) -> bool:
        """Return True if we need to handle empty lists and/or nulls."""
        return self.empty_as_null or self.keep_nulls


class UniqueOptions(Immutable):
    __slots__ = ("keep", "maintain_order")
    keep: UniqueKeepStrategy
    maintain_order: bool

    @staticmethod
    def default() -> UniqueOptions:  # pragma: no cover
        return UniqueOptions(keep="any", maintain_order=False)

    @staticmethod
    def parse(
        keep: UniqueKeepStrategy, /, *, maintain_order: bool
    ) -> UniqueOptions:  # pragma: no cover
        if keep in {"any", "first", "last", "none"}:
            return UniqueOptions(keep=keep, maintain_order=maintain_order)
        msg = f"Only the following keep strategies are supported: {get_args(UniqueKeepStrategy)}; found '{keep}'."
        raise NotImplementedError(msg)


class VConcatOptions(Immutable):
    __slots__ = ("diagonal", "maintain_order", "to_supertypes")
    diagonal: bool
    """True for `how="diagonal"`"""

    to_supertypes: bool
    """True for [`"*_relaxed"` variants]

    [`"*_relaxed"` variants]: https://github.com/narwhals-dev/narwhals/pull/3191#issuecomment-3389117044
    """

    maintain_order: bool
    """True when using `concat`, False when using [`union`].

    [`union`]: https://github.com/pola-rs/polars/pull/24298
    """

    @staticmethod
    def from_how(
        how: Literal["vertical", "diagonal", "vertical_relaxed", "diagonal_relaxed"],
        /,
        *,
        maintain_order: bool = True,
    ) -> VConcatOptions:  # pragma: no cover
        return VConcatOptions(
            diagonal=(how.startswith("diagonal")),
            to_supertypes=how.endswith("relaxed"),
            maintain_order=maintain_order,
        )


JoinStrategyT = TypeVar("JoinStrategyT", bound=JoinStrategy)
JoinStrategyT_co = TypeVar(
    "JoinStrategyT_co", bound=JoinStrategy, default=JoinStrategy, covariant=True
)


class JoinOptions(Immutable, Generic[JoinStrategyT_co]):
    __slots__ = ("how", "suffix")
    how: JoinStrategyT_co  # type: ignore[misc]
    suffix: str

    @staticmethod
    def default() -> JoinOptions:  # pragma: no cover
        return JoinOptions(how="inner", suffix="_right")

    @staticmethod
    def parse(how: JoinStrategyT, suffix: str) -> JoinOptions[JoinStrategyT]:
        if how in {"inner", "left", "full", "cross", "anti", "semi"}:
            return JoinOptions(how=how, suffix=suffix)
        msg = f"Only the following join strategies are supported: {get_args(JoinStrategy)}; found '{how}'."
        raise NotImplementedError(msg)

    def normalize_on(
        self: JoinOptions[NonCrossJoinStrategy],
        on: OneOrIterable[str] | None,
        left_on: OneOrIterable[str] | None,
        right_on: OneOrIterable[str] | None,
        /,
    ) -> tuple[Seq[str], Seq[str]]:
        """Reduce the 3 potential key (`*on`) arguments to 2.

        Ensures the keys spelling is compatible with the join strategy.
        """
        if on is None:
            if left_on is None or right_on is None:
                msg = f"Either (`left_on` and `right_on`) or `on` keys should be specified for {self.how}."
                raise ValueError(msg)
            left_on = ensure_seq_str(left_on)
            right_on = ensure_seq_str(right_on)
            if len(left_on) != len(right_on):
                msg = "`left_on` and `right_on` must have the same length."
                raise ValueError(msg)
            return left_on, right_on
        if left_on is not None or right_on is not None:
            msg = f"If `on` is specified, `left_on` and `right_on` should be None for {self.how}."
            raise ValueError(msg)
        on = ensure_seq_str(on)
        return on, on


class JoinAsofBy(Immutable):  # pragma: no cover
    __slots__ = ("left_by", "right_by")
    left_by: Seq[str]
    right_by: Seq[str]

    @staticmethod
    def parse(
        by_left: str | Sequence[str] | None,
        by_right: str | Sequence[str] | None,
        by: str | Sequence[str] | None,
    ) -> JoinAsofBy:
        if by is None:
            if by_left and by_right:
                left_by = ensure_seq_str(by_left)
                right_by = ensure_seq_str(by_right)
                if len(left_by) != len(right_by):
                    msg = "`by_left` and `by_right` must have the same length."
                    raise ValueError(msg)
                return JoinAsofBy(left_by=left_by, right_by=right_by)
            msg = (
                "Can not specify only `by_left` or `by_right`, you need to specify both."
            )
            raise ValueError(msg)
        if by_left or by_right:
            msg = "If `by` is specified, `by_left` and `by_right` should be None."
            raise ValueError(msg)
        by_ = ensure_seq_str(by)  # pragma: no cover
        return JoinAsofBy(left_by=by_, right_by=by_)  # pragma: no cover


class JoinAsofOptions(Immutable):  # pragma: no cover
    # polars is being quite inconsistent here
    # - LazyFrame.join_asof
    # - AsofJoinStrategy
    # - JoinType::AsOf(Box<AsOfOptions>)
    # - #[cfg(feature = "asof_join")]
    # - join::asof::
    #   - AsofStrategy
    #   - trait AsofJoin (fn _join_asof)
    __slots__ = ("by", "strategy", "suffix")
    by: JoinAsofBy | None
    strategy: JoinAsofStrategy
    suffix: str

    @staticmethod
    def parse(
        by_left: str | Sequence[str] | None = None,
        by_right: str | Sequence[str] | None = None,
        by: str | Sequence[str] | None = None,
        strategy: JoinAsofStrategy = "backward",
        suffix: str = "_right",
    ) -> JoinAsofOptions:
        if by_left or by_right or by:
            opts = JoinAsofBy.parse(by_left, by_right, by)
        else:
            opts = None
        if strategy in {"backward", "forward", "nearest"}:
            return JoinAsofOptions(by=opts, strategy=strategy, suffix=suffix)
        msg = f"Only the following join strategies are supported: {get_args(AsofJoinStrategy)}; found '{strategy}'."
        raise NotImplementedError(msg)

    @staticmethod
    def normalize_on(
        left_on: str | None, right_on: str | None, on: str | None
    ) -> tuple[str, str]:
        """Reduce the 3 potential `join_asof` (`*on`) arguments to 2."""
        if on is None:
            if left_on is None or right_on is None:
                msg = (
                    "Either (`left_on` and `right_on`) or `on` keys should be specified."
                )
                raise ValueError(msg)
            return left_on, right_on
        if left_on is not None or right_on is not None:
            msg = "If `on` is specified, `left_on` and `right_on` should be None."
            raise ValueError(msg)
        return on, on


class UnpivotOptions(Immutable):
    __slots__ = ("value_name", "variable_name")
    variable_name: str
    value_name: str

    @staticmethod
    def default() -> UnpivotOptions:  # pragma: no cover
        return UnpivotOptions(variable_name="variable", value_name="value")
