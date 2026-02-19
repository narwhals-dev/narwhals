"""General functions that aren't namespaced."""

from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals._plan.dtypes_mapper as dtm
from narwhals._plan._dtype import ResolveDType
from narwhals._plan._function import Function, HorizontalFunction
from narwhals._plan.exceptions import hist_bins_monotonic_error
from narwhals._plan.options import FunctionFlags, FunctionOptions

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from typing import Any

    from _typeshed import ConvertibleToInt
    from typing_extensions import Self

    from narwhals._plan._expr_ir import ExprIR
    from narwhals._plan.expressions.expr import AnonymousExpr, FunctionExpr, RollingExpr
    from narwhals._plan.options import EWMOptions, RankOptions, RollingOptionsFixedWindow
    from narwhals._plan.schema import FrozenSchema
    from narwhals._plan.typing import Seq, Udf
    from narwhals.dtypes import DType
    from narwhals.typing import FillNullStrategy

# NOTE: `pylance` (via `pyright`) doesn't show `__init_subclass__` usage in *Find All References*,
# but hopefully `ty` will https://github.com/astral-sh/ty/issues/1777#issuecomment-3618906859
# These aliases work around that limitation (+ are shorter 🙂)
aggregation = FunctionOptions.aggregation
elementwise = FunctionOptions.elementwise
length_preserving = FunctionOptions.length_preserving
row_separable = FunctionOptions.row_separable
map_first = ResolveDType.function.map_first
same_dtype = ResolveDType.function.same_dtype


# fmt: off
class _SameDType(Function, dtype=same_dtype()): ...
class Abs(_SameDType, options=elementwise): ...
class NullCount(Function, options=aggregation, dtype=dtm.IDX_DTYPE): ...
class Exp(Function, options=elementwise, dtype=map_first(dtm.float_dtype)): ...
class Sqrt(Function, options=elementwise, dtype=map_first(dtm.numeric_to_float_dtype_coerce_decimal)): ...
class Ceil(_SameDType, options=elementwise): ...
class Floor(_SameDType, options=elementwise): ...
class DropNulls(_SameDType, options=row_separable): ...
class ModeAll(_SameDType): ...
class ModeAny(_SameDType, options=aggregation): ...
class Kurtosis(Function, options=aggregation, dtype=dtm.F64): ...
class Skew(Function, options=aggregation, dtype=dtm.F64): ...
class Clip(_SameDType, options=elementwise):
    def unwrap_input(self, node: FunctionExpr[Self], /) -> tuple[ExprIR, ExprIR, ExprIR]:
        expr, lower_bound, upper_bound = node.input
        return expr, lower_bound, upper_bound
class ClipLower(_SameDType, options=elementwise):
    def unwrap_input(self, node: FunctionExpr[Self], /) -> tuple[ExprIR, ExprIR]:
        expr, lower_bound = node.input
        return expr, lower_bound
class ClipUpper(_SameDType, options=elementwise):
    def unwrap_input(self, node: FunctionExpr[Self], /) -> tuple[ExprIR, ExprIR]:
        expr, upper_bound = node.input
        return expr, upper_bound
class CumAgg(Function, options=length_preserving):
    __slots__ = ("reverse",)
    reverse: bool
class CumCount(CumAgg, dtype=dtm.IDX_DTYPE): ...
class CumMin(CumAgg, dtype=same_dtype()): ...
class CumMax(CumAgg, dtype=same_dtype()): ...
class CumProd(CumAgg, dtype=map_first(dtm.cum_prod_dtype)): ...
class CumSum(CumAgg, dtype=map_first(dtm.cum_sum_dtype)): ...
class RollingWindow(Function, options=length_preserving):
    __slots__ = ("options",)
    options: RollingOptionsFixedWindow

    def to_function_expr(self, *inputs: ExprIR) -> RollingExpr[Self]:
        from narwhals._plan.expressions.expr import RollingExpr

        options = self.function_options
        return RollingExpr(input=inputs, function=self, options=options)
class RollingSum(RollingWindow, dtype=map_first(dtm.sum_dtype)): ...
class RollingMean(RollingWindow, dtype=map_first(dtm.moment_dtype)): ...
class RollingVar(RollingWindow, dtype=map_first(dtm.var_dtype)): ...
class RollingStd(RollingWindow, dtype=map_first(dtm.moment_dtype)): ...
class Diff(Function, options=length_preserving, dtype=map_first(dtm.diff_dtype)): ...
class Unique(_SameDType): ...
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
    def resolve_dtype(self, node: FunctionExpr[Self], schema: FrozenSchema, /) -> DType:
        # NOTE: There are 6 supertype pairs (that we support) that could produce `Float32`
        # Otherwise, it is always `Float64`
        if dtm.F32 in {e.resolve_dtype(schema) for e in node.input}:
            msg = f"{self!r} is not yet supported when inputs contain a {dtm.F32!r} dtype.\n"
            "This operation requires https://github.com/narwhals-dev/narwhals/pull/3396"
            raise NotImplementedError(msg)
        return dtm.F64


class Hist(
    Function,
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


class Log(Function, options=elementwise, dtype=map_first(dtm.float_dtype)):
    __slots__ = ("base",)
    base: float


class Pow(Function, options=elementwise):
    """N-ary (base, exponent)."""

    def unwrap_input(self, node: FunctionExpr[Self], /) -> tuple[ExprIR, ExprIR]:
        base, exponent = node.input
        return base, exponent

    def resolve_dtype(self, node: FunctionExpr[Self], schema: FrozenSchema, /) -> DType:
        base = node.input[0].resolve_dtype(schema)
        if base.is_integer() and (exp := node.input[1].resolve_dtype(schema)).is_float():
            return exp
        return base


class FillNull(Function, options=elementwise):
    """N-ary (expr, value)."""

    def unwrap_input(self, node: FunctionExpr[Self], /) -> tuple[ExprIR, ExprIR]:
        expr, value = node.input
        return expr, value

    # TODO @dangotbanned: `map_to_supertype`
    def resolve_dtype(self, node: FunctionExpr[Self], schema: FrozenSchema, /) -> DType:
        expr, value = (e.resolve_dtype(schema) for e in node.input)
        if expr != value:
            msg = f"{self!r} is currently only supported when the dtype of the expression {expr!r} matches the fill value {value!r}.\n"
            "This operation requires https://github.com/narwhals-dev/narwhals/pull/3396"
            raise NotImplementedError(msg)
        return expr


class FillNan(_SameDType, options=elementwise):
    """N-ary (expr, value)."""

    def unwrap_input(self, node: FunctionExpr[Self], /) -> tuple[ExprIR, ExprIR]:
        expr, value = node.input
        return expr, value


class FillNullWithStrategy(_SameDType):
    __slots__ = ("limit", "strategy")
    strategy: FillNullStrategy
    limit: int | None


class Shift(_SameDType, options=length_preserving):
    __slots__ = ("n",)
    n: int


class Rank(
    Function, dtype=lambda f: dtm.F64 if f.options.method == "average" else dtm.IDX_DTYPE
):
    __slots__ = ("options",)
    options: RankOptions


class Round(_SameDType, options=elementwise):
    __slots__ = ("decimals",)
    decimals: int


class EwmMean(
    Function,
    options=length_preserving,
    dtype=map_first(dtm.numeric_to_float_dtype_coerce_decimal),
):
    __slots__ = ("options",)
    options: EWMOptions


# TODO @dangotbanned: (partial)
# Need to run a sample of `new` through `nwp.common.py_to_narwhals_dtype`
class ReplaceStrict(Function, options=elementwise):
    __slots__ = ("new", "old", "return_dtype")
    old: Seq[Any]
    new: Seq[Any]
    return_dtype: DType | None

    def resolve_dtype(self, node: FunctionExpr[Self], schema: FrozenSchema, /) -> DType:
        if dtype := self.return_dtype:
            return dtype
        # NOTE: polars would use the dtype of `new` here
        # https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L773
        # https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L777
        # https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L781
        return super().resolve_dtype(node, schema)


# TODO @dangotbanned: (partial) `get_supertype(new, default)`
# Similar to `ReplaceStrict.resolve_dtype`
# https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L780
class ReplaceStrictDefault(ReplaceStrict):
    def unwrap_input(self, node: FunctionExpr[Self], /) -> tuple[ExprIR, ExprIR]:
        expr, default = node.input
        return expr, default


class GatherEvery(_SameDType):
    __slots__ = ("n", "offset")
    n: int
    offset: int


class MapBatches(Function):
    __slots__ = ("function", "is_elementwise", "return_dtype", "returns_scalar")
    function: Udf
    return_dtype: DType | None
    is_elementwise: bool
    returns_scalar: bool

    @property
    def function_options(self) -> FunctionOptions:
        options = super().function_options
        if self.is_elementwise:
            options = options.with_elementwise()
        if self.returns_scalar:
            options = options.with_flags(FunctionFlags.RETURNS_SCALAR)
        return options

    def to_function_expr(self, *inputs: ExprIR) -> AnonymousExpr:
        from narwhals._plan.expressions.expr import AnonymousExpr

        options = self.function_options
        return AnonymousExpr(input=inputs, function=self, options=options)


class SampleN(_SameDType):
    __slots__ = ("n", "seed", "with_replacement")
    n: int
    with_replacement: bool
    seed: int | None


class SampleFrac(_SameDType):
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
