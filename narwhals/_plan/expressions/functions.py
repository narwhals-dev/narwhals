"""General functions that aren't namespaced.

TODO @dangotbanned: Rename module and use the current name to re-export all public functions
- Export members of `ranges` & `boolean` to that namespace
- `{categorical,lists,strings,struct,temporal}` -> `{cat,list,str,struct,dt}`
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals._plan.dtypes_mapper as dtm
from narwhals._plan._dtype import ResolveDType
from narwhals._plan._flags import FunctionFlags
from narwhals._plan._function import (
    BinaryFunction,
    Function,
    HorizontalFunction,
    TernaryFunction,
    UnaryFunction,
)
from narwhals._plan.exceptions import hist_bins_monotonic_error

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from typing import Any

    from _typeshed import ConvertibleToInt
    from typing_extensions import Self

    from narwhals._plan.expressions import AnonymousExpr, FunctionExpr
    from narwhals._plan.options import (
        EWMOptions,
        RankOptions,
        RollingOptions,
        RollingVarOptions,
    )
    from narwhals._plan.schema import FrozenSchema
    from narwhals._plan.typing import Seq, Udf
    from narwhals.dtypes import DType
    from narwhals.typing import FillNullStrategy

# NOTE: `pylance` (via `pyright`) doesn't show `__init_subclass__` usage in *Find All References*,
# but hopefully `ty` will https://github.com/astral-sh/ty/issues/1777#issuecomment-3618906859
# These aliases work around that limitation (+ are shorter 🙂)
AGGREGATION = FunctionFlags.AGGREGATION
ELEMENTWISE = FunctionFlags.ELEMENTWISE
LENGTH_PRESERVING = FunctionFlags.LENGTH_PRESERVING
ROW_SEPARABLE = FunctionFlags.ROW_SEPARABLE
map_first = ResolveDType.function.map_first
same_dtype = ResolveDType.function.same_dtype


# fmt: off
class _UnarySameDType(UnaryFunction, dtype=same_dtype()): ...
class Abs(_UnarySameDType, flags=ELEMENTWISE): ...
class NullCount(UnaryFunction, flags=AGGREGATION, dtype=dtm.IDX_DTYPE): ...
class Exp(UnaryFunction, flags=ELEMENTWISE, dtype=map_first(dtm.float_dtype)): ...
class Sqrt(UnaryFunction, flags=ELEMENTWISE, dtype=map_first(dtm.numeric_to_float_dtype_coerce_decimal)): ...
class Ceil(_UnarySameDType, flags=ELEMENTWISE): ...
class Floor(_UnarySameDType, flags=ELEMENTWISE): ...
class DropNulls(_UnarySameDType, flags=ROW_SEPARABLE): ...
class ModeAll(_UnarySameDType): ...
class ModeAny(_UnarySameDType, flags=AGGREGATION): ...
class Kurtosis(UnaryFunction, flags=AGGREGATION, dtype=dtm.F64): ...
class Skew(UnaryFunction, flags=AGGREGATION, dtype=dtm.F64): ...
class Clip(TernaryFunction, dtype=same_dtype(), flags=ELEMENTWISE): ...
class ClipLower(BinaryFunction, dtype=same_dtype(), flags=ELEMENTWISE): ...
class ClipUpper(BinaryFunction, dtype=same_dtype(), flags=ELEMENTWISE): ...
class CumAgg(UnaryFunction, flags=LENGTH_PRESERVING):
    __slots__ = ("reverse",)
    reverse: bool
class CumCount(CumAgg, dtype=dtm.IDX_DTYPE): ...
class CumMin(CumAgg, dtype=same_dtype()): ...
class CumMax(CumAgg, dtype=same_dtype()): ...
class CumProd(CumAgg, dtype=map_first(dtm.cum_prod_dtype)): ...
class CumSum(CumAgg, dtype=map_first(dtm.cum_sum_dtype)): ...
class RollingWindow(UnaryFunction, flags=LENGTH_PRESERVING):
    __slots__ = ("options",)
    options: RollingOptions
class RollingSum(RollingWindow, dtype=map_first(dtm.sum_dtype)): ...
class RollingMean(RollingWindow, dtype=map_first(dtm.moment_dtype)): ...
class _RollingVarStd(RollingWindow):
    options: RollingVarOptions
class RollingVar(_RollingVarStd, dtype=map_first(dtm.var_dtype)): ...
class RollingStd(_RollingVarStd, dtype=map_first(dtm.moment_dtype)): ...
class Diff(UnaryFunction, flags=LENGTH_PRESERVING, dtype=map_first(dtm.diff_dtype)): ...
class Unique(_UnarySameDType): ...
# TODO @dangotbanned: `map_to_supertype` (`*Horizontal`)
# - https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L45
# - https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L402-L420
# - https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L410-L420
# - https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L806-L830
class SumHorizontal(HorizontalFunction): ...
class MinHorizontal(HorizontalFunction): ...
class MaxHorizontal(HorizontalFunction): ...
class Coalesce(HorizontalFunction): ...
# fmt: on
class MeanHorizontal(HorizontalFunction):
    # TODO @dangotbanned: `map_to_supertype`
    def resolve_dtype(
        self, node: FunctionExpr[Self], schema: FrozenSchema, /
    ) -> DType:  # pragma: no cover
        # NOTE: There are 6 supertype pairs (that we support) that could produce `Float32`
        # Otherwise, it is always `Float64`
        if dtm.F32 in {e.resolve_dtype(schema) for e in node.input}:
            msg = f"{self!r} is not yet supported when inputs contain a {dtm.F32!r} dtype.\n"
            "This operation requires https://github.com/narwhals-dev/narwhals/pull/3396"
            raise NotImplementedError(msg)
        return dtm.F64


class Hist(
    UnaryFunction,
    # https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L220-L243
    dtype=lambda f: (
        dtm.Struct({"breakpoint": dtm.F64, "count": dtm.IDX_DTYPE})
        if f.include_breakpoint
        else dtm.IDX_DTYPE
    ),
):
    __slots__ = ("include_breakpoint",)
    include_breakpoint: bool

    def __repr__(self) -> str:
        return "hist"

    # NOTE: These constructors provide validation + defaults, and avoid
    # repeating on every `__init__` afterwards
    # They're also more widely defined to what will work at runtime
    @staticmethod
    def from_bins(
        bins: Iterable[float], /, *, include_breakpoint: bool = False
    ) -> HistBins:
        bins = tuple(bins)
        for i in range(1, len(bins)):
            if bins[i - 1] >= bins[i]:
                raise hist_bins_monotonic_error(bins)
        return HistBins(bins=bins, include_breakpoint=include_breakpoint)

    @staticmethod
    def from_bin_count(
        count: ConvertibleToInt = 10, /, *, include_breakpoint: bool = False
    ) -> HistBinCount:
        return HistBinCount(bin_count=int(count), include_breakpoint=include_breakpoint)

    @property
    def empty_data(self) -> Mapping[str, Iterable[Any]]:
        # NOTE: Need to adapt for `include_category`
        return (
            {"breakpoint": [], "count": []} if self.include_breakpoint else {"count": []}
        )


class HistBins(Hist):
    __slots__ = ("bins",)
    bins: Seq[float]


class HistBinCount(Hist):
    __slots__ = ("bin_count",)
    bin_count: int


class Log(UnaryFunction, flags=ELEMENTWISE, dtype=map_first(dtm.float_dtype)):
    __slots__ = ("base",)
    base: float


class Pow(BinaryFunction, flags=ELEMENTWISE):
    def resolve_dtype(
        self, node: FunctionExpr[Self], schema: FrozenSchema, /
    ) -> DType:  # pragma: no cover
        base = node.input[0].resolve_dtype(schema)
        if base.is_integer() and (exp := node.input[1].resolve_dtype(schema)).is_float():
            return exp
        return base


class FillNull(BinaryFunction, flags=ELEMENTWISE):
    # TODO @dangotbanned: `map_to_supertype`
    def resolve_dtype(
        self, node: FunctionExpr[Self], schema: FrozenSchema, /
    ) -> DType:  # pragma: no cover
        expr, value = (e.resolve_dtype(schema) for e in node.input)
        if expr != value:
            msg = f"{self!r} is currently only supported when the dtype of the expression {expr!r} matches the fill value {value!r}.\n"
            "This operation requires https://github.com/narwhals-dev/narwhals/pull/3396"
            raise NotImplementedError(msg)
        return expr


class FillNullWithStrategy(_UnarySameDType):
    __slots__ = ("limit", "strategy")
    strategy: FillNullStrategy
    limit: int | None


class Shift(_UnarySameDType, flags=LENGTH_PRESERVING):
    __slots__ = ("n",)
    n: int


class Rank(
    UnaryFunction,
    dtype=lambda f: dtm.F64 if f.options.method == "average" else dtm.IDX_DTYPE,
):
    __slots__ = ("options",)
    options: RankOptions


class Round(_UnarySameDType, flags=ELEMENTWISE):
    __slots__ = ("decimals",)
    decimals: int


class EwmMean(
    UnaryFunction,
    flags=LENGTH_PRESERVING,
    dtype=map_first(dtm.numeric_to_float_dtype_coerce_decimal),
):
    __slots__ = ("options",)
    options: EWMOptions


# TODO @dangotbanned: Finish partial `replace_strict` resolve dtype
@ResolveDType.function.visitor
def _replace_strict_dtype(
    self: ReplaceStrict | ReplaceStrictDefault, /
) -> DType:  # pragma: no cover
    """(Partial) impl of `resolve_dtype`.

    - `ReplaceStrict`
      - `new` is used when missing `return_dtype` ([1], [2], [3])

      - Need to run a sample of `new` through `nwp.common.py_to_narwhals_dtype`
    - `ReplaceStrictDefault`
      - Builds on the above, then needs `get_supertype(new, default)` ([4])

    [1]: https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L773
    [2]: https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L777
    [3]: https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L781
    [4]: https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L780
    """
    if dtype := self.return_dtype:
        return dtype
    msg = f"`{type(self).__name__}.resolve_dtype()` is not yet fully implemented"
    raise NotImplementedError(msg)


class ReplaceStrict(UnaryFunction, flags=ELEMENTWISE, dtype=_replace_strict_dtype):
    __slots__ = ("new", "old", "return_dtype")
    old: Seq[Any]
    new: Seq[Any]
    return_dtype: DType | None


class ReplaceStrictDefault(
    BinaryFunction, flags=ELEMENTWISE, dtype=_replace_strict_dtype
):
    __slots__ = ("new", "old", "return_dtype")
    old: Seq[Any]
    new: Seq[Any]
    return_dtype: DType | None


class GatherEvery(_UnarySameDType):
    __slots__ = ("n", "offset")
    n: int
    offset: int


class MapBatches(UnaryFunction):
    __slots__ = ("flags", "function", "return_dtype")
    function: Udf
    return_dtype: DType | None
    flags: FunctionFlags

    @staticmethod
    def from_udf(
        function: Udf,
        return_dtype: DType | None,
        *,
        is_elementwise: bool,
        returns_scalar: bool,
    ) -> MapBatches:
        flags = Function.__function_flags__
        if is_elementwise:
            flags |= ELEMENTWISE
        if returns_scalar:
            flags |= AGGREGATION
        return MapBatches(function=function, return_dtype=return_dtype, flags=flags)

    def is_elementwise(self) -> bool:
        return self.flags.is_elementwise()

    is_length_preserving = is_elementwise

    @classmethod
    def __function_expr__(cls) -> type[AnonymousExpr]:
        from narwhals._plan.expressions import AnonymousExpr

        return AnonymousExpr

    def resolve_dtype(
        self, node: FunctionExpr[Self], schema: FrozenSchema, /
    ) -> DType:  # pragma: no cover
        if dtype := self.return_dtype:
            return dtype
        return super().resolve_dtype(node, schema)


class SampleN(_UnarySameDType):
    __slots__ = ("n", "seed", "with_replacement")
    n: int
    with_replacement: bool
    seed: int | None


class SampleFrac(_UnarySameDType):
    __slots__ = ("fraction", "seed", "with_replacement")
    fraction: float
    with_replacement: bool
    seed: int | None


def sample(
    n: int | None = None,
    *,
    fraction: float | None = None,
    with_replacement: bool = False,
    seed: int | None = None,
) -> SampleFrac | SampleN:
    if n is not None and fraction is not None:
        msg = "cannot specify both `n` and `fraction`"
        raise ValueError(msg)
    if fraction is not None:
        return SampleFrac(fraction=fraction, with_replacement=with_replacement, seed=seed)
    return SampleN(n=1 if n is None else n, with_replacement=with_replacement, seed=seed)
