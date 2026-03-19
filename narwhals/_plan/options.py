from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Generic, Literal, final, get_args

from narwhals._plan._immutable import Immutable
from narwhals._plan.common import ensure_seq_str
from narwhals._typing_compat import TypeVar
from narwhals._utils import Implementation, ensure_type
from narwhals.exceptions import InvalidOperationError
from narwhals.typing import AsofJoinStrategy, JoinStrategy, UniqueKeepStrategy

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import TypedDict

    import pyarrow.compute as pc

    from narwhals._plan.typing import NonCrossJoinStrategy, OneOrIterable, Seq
    from narwhals._typing import Backend
    from narwhals.typing import AsofJoinStrategy as JoinAsofStrategy, RankMethod

    # TODO @dangotbanned: Replace with `dict[Literal["descending", "nulls_last"], Seq[bool]]` after bumping mypy
    # to include https://github.com/python/mypy/pull/20416
    class _SortOptions(TypedDict):
        descending: bool | Seq[bool]
        nulls_last: bool | Seq[bool]


_OBJ_SETATTR = object.__setattr__


# TODO @dangotbanned: Finish content and move most to `FunctionOptions`
# TODO @dangotbanned: Class doc?
class FunctionFlags(enum.Flag):
    # TODO @dangotbanned: Figure out new default + remove
    ALLOW_GROUP_AWARE = 1 << 0
    """Raise if use in group by

    ## History
    - Flag removed in [#23690], but left a vestigial `FunctionOptions.groupwise()`
    - Acts as a default

    [#23690]: https://github.com/pola-rs/polars/pull/23690
    """

    REDUCE_EXPANSION = 1 << 2
    """Use different semantics when expanding selectors.

    Upstream this is named [`INPUT_WILDCARD_EXPANSION`].

    ## Examples
    Say we have the following schema:
    >>> from tests.plan.utils import Frame
    >>> import narwhals._plan as nw

    >>> df = Frame.from_names("a", "b", "c")
    >>> df.schema
    Schema({'a': Int64, 'b': Int64, 'c': Int64})

    This kind of expansion is used for inputs to horizontal functions:
    >>> before = nw.sum_horizontal(nw.all())
    >>> (reduced,) = df.project(before)
    >>> before._ir
    ncs.all().sum_horizontal()
    >>> reduced
    a=col('a').sum_horizontal([col('b'), col('c')])

    Whereas the more common form of expansion produces multiple outputs:
    >>> before = nw.all().clip("b")
    >>> before._ir
    ncs.all().clip_lower([col('b')])
    >>> df.project(before)  # doctest: +NORMALIZE_WHITESPACE
    (a=col('a').clip_lower([col('b')]),
     b=col('b').clip_lower([col('b')]),
     c=col('c').clip_lower([col('b')]))

    [`INPUT_WILDCARD_EXPANSION`]: https://github.com/pola-rs/polars/blob/b6ae11535a9a45a442446ad13f687616ca97ee95/crates/polars-plan/src/plans/options.rs#L66-L76
    """

    # TODO @dangotbanned: Doc (full rewrite)
    # 1. AFAIK, using "unit" in this way isn't common in python
    # 2. These might have been `Function`s in an early version,
    #    but they aren't now (find relevant examples)
    #    - `sum`, `min` are `AggExpr`
    #    - `head_1` probably means `Gather` or `Slice`
    AGGREGATION = 1 << 3
    """Automatically explode on unit length if it ran as final aggregation.

    this is the case for aggregations like sum, min, covariance etc.
    We need to know this because we cannot see the difference between
    the following functions based on the output type and number of elements:

        x = [1, 2, 3]

        head_1(x) -> [1]
        sum(x) -> [4]
    """

    ROW_SEPARABLE = 1 << 6
    """Does not depend on the context of surrounding rows.

    Only `drop_nulls`, `drop_nans`.

    ## Important
    Row-separable functions are partially defined by what they are **not**:

        ELEMENTWISE = ROW_SEPARABLE | LENGTH_PRESERVING
        #             ✔️              ❌

    This definition means they change the length of input columns.

    But this property **does not** extend to elementwise.

    ## History
    ~~`*_range`~~ since [#26549](https://github.com/pola-rs/polars/pull/26549)
    """

    LENGTH_PRESERVING = 1 << 7
    """Does not change the length of input columns.

    Includes `rolling_*`, `cum_*`, `shift`.

    ## Important
    Length-preserving functions are partially defined by what they are **not**:

        ELEMENTWISE = ROW_SEPARABLE | LENGTH_PRESERVING
        #             ❌              ✔️

    This definition means they depend on the context of surrounding rows.

    But this property **does not** extend to elementwise.
    """

    ELEMENTWISE = ROW_SEPARABLE | LENGTH_PRESERVING
    HORIZONTAL = REDUCE_EXPANSION | ELEMENTWISE

    def __str__(self) -> str:
        name = self.name or "<FUNCTION_FLAGS_UNKNOWN>"
        return name.replace("|", " | ")


# NOTE: `FunctionFlag`s get some heavy use, but always within the context of a `FunctionOptions`
# If `FunctionOptions` is in scope, these aliases are too and save a bunch of lookups
_DEFAULT = _ALLOW_GROUP_AWARE = FunctionFlags.ALLOW_GROUP_AWARE
_REDUCE_EXPANSION = FunctionFlags.REDUCE_EXPANSION
_AGGREGATION = FunctionFlags.AGGREGATION
_ROW_SEPARABLE = FunctionFlags.ROW_SEPARABLE
_LENGTH_PRESERVING = FunctionFlags.LENGTH_PRESERVING
_ELEMENTWISE = FunctionFlags.ELEMENTWISE
_HORIZONTAL = FunctionFlags.HORIZONTAL
_INVALID = FunctionFlags.AGGREGATION | FunctionFlags.LENGTH_PRESERVING


# TODO @dangotbanned: Explain `FunctionOptions` (class)
# TODO @dangotbanned: Explain `flags` (attr)
# TODO @dangotbanned: Decide on how to split `FunctionFlags` docs
# TODO @dangotbanned: Explain `FunctionOptions` (staticmethods)
# TODO @dangotbanned: Explain `FunctionOptions` (guards)
@final
class FunctionOptions(Immutable):
    """_summary_.

    https://github.com/pola-rs/polars/blob/3fd7ecc5f9de95f62b70ea718e7e5dbf951b6d1c/crates/polars-plan/src/plans/options.rs
    """

    __slots__ = ("flags",)
    flags: FunctionFlags
    """_summary_."""

    def __str__(self) -> str:
        return f"{type(self).__name__}(flags='{self.flags}')"

    def is_elementwise(self) -> bool:
        """_summary_."""
        return _ELEMENTWISE in self.flags

    def is_reduce_expansion(self) -> bool:
        """_summary_."""
        return _REDUCE_EXPANSION in self.flags

    def is_length_preserving(self) -> bool:
        """_summary_."""
        return _LENGTH_PRESERVING in self.flags  # pragma: no cover

    def is_row_separable(self) -> bool:
        """_summary_."""
        return _ROW_SEPARABLE in self.flags

    def is_aggregation(self) -> bool:
        """_summary_."""
        return _AGGREGATION in self.flags

    @staticmethod
    def aggregation() -> FunctionOptions:
        """_summary_.

        Mutually exclusive with `length_preserving`
        """
        return FunctionOptions._default_with_flags(_AGGREGATION)

    @staticmethod
    def elementwise() -> FunctionOptions:
        """_summary_."""
        return FunctionOptions._default_with_flags(_ELEMENTWISE)

    @staticmethod
    def groupwise() -> FunctionOptions:
        """_summary_."""
        obj = FunctionOptions.__new__(FunctionOptions)
        _OBJ_SETATTR(obj, "flags", _ALLOW_GROUP_AWARE)
        return obj

    @staticmethod
    def horizontal() -> FunctionOptions:
        """_summary_."""
        return FunctionOptions._default_with_flags(_HORIZONTAL)

    @staticmethod
    def length_preserving() -> FunctionOptions:
        """Does not change the length of input columns.

        Depends on the context of surrounding rows.

        Mutually exclusive with `aggregation`
        """
        return FunctionOptions._default_with_flags(_LENGTH_PRESERVING)

    @staticmethod
    def row_separable() -> FunctionOptions:
        """Does not depend on the context of surrounding rows.

        Changes the length of input columns.
        """
        return FunctionOptions._default_with_flags(_ROW_SEPARABLE)

    def with_flags(self, flags: FunctionFlags, /) -> FunctionOptions:
        """Create a new set of options, combining `flags` with self.

        Ensures `flags` is compatible with current expression.
        """
        new_flags = self.flags | flags
        if _INVALID in new_flags:
            msg = "A function cannot both return a scalar and preserve length, they are mutually exclusive."
            raise TypeError(msg)
        obj = FunctionOptions.__new__(FunctionOptions)
        _OBJ_SETATTR(obj, "flags", new_flags)
        return obj

    def with_udf(self, *, is_elementwise: bool, returns_scalar: bool) -> FunctionOptions:
        """Special-case of `with_flags` for inputs from `map_batches`."""
        opts = self
        if is_elementwise:
            opts = self.with_flags(_ELEMENTWISE)
        if returns_scalar:
            opts = opts.with_flags(_AGGREGATION)
        return opts

    # TODO @dangotbanned: Remove at the same time as `ALLOW_GROUP_AWARE`
    @staticmethod
    def _default_with_flags(flags: FunctionFlags, /) -> FunctionOptions:
        obj = FunctionOptions.__new__(FunctionOptions)
        _OBJ_SETATTR(obj, "flags", _DEFAULT | flags)
        return obj

    default = groupwise


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

    def to_polars(self, by: Sequence[str]) -> _SortOptions:
        """[`extend_bool`] doesn't broadcast length 1 sequences, so we do it here.

        [`extend_bool`]: https://github.com/pola-rs/polars/blob/b8bfb07a4a37a8d449d6d1841e345817431142df/py-polars/polars/_utils/various.py#L580-L594
        """
        len_by = len(by)
        desc, nulls = self.descending, self.nulls_last
        if len_by != 1:
            desc = desc if len(desc) != 1 else desc * len_by
            nulls = nulls if len(nulls) != 1 else nulls * len_by
        return {"descending": desc, "nulls_last": nulls}


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
    def parse(
        how: JoinStrategyT, suffix: str
    ) -> JoinOptions[JoinStrategyT]:  # pragma: no cover
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
    ) -> tuple[Seq[str], Seq[str]]:  # pragma: no cover
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
