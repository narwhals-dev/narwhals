"""General functions that aren't namespaced."""

from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals._plan.dtypes_mapper as dtm
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


class _SameDType(Function):
    def _resolve_dtype(self, schema: FrozenSchema, node: FunctionExpr[Function]) -> DType:
        return node.input[0]._resolve_dtype(schema)


class _F64DType(Function):
    def _resolve_dtype(self, schema: FrozenSchema, node: FunctionExpr[Function]) -> DType:
        return dtm.F64


class _NumericToFloatDType(Function):
    def _resolve_dtype(self, schema: FrozenSchema, node: FunctionExpr[Function]) -> DType:
        return dtm.numeric_to_float_dtype_coerce_decimal(
            node.input[0]._resolve_dtype(schema)
        )


# TODO @dangotbanned: CumProd, CumSum
class CumAgg(Function, options=FunctionOptions.length_preserving):
    __slots__ = ("reverse",)
    reverse: bool

    def _resolve_dtype(self, schema: FrozenSchema, node: FunctionExpr[Function]) -> DType:
        tp = type(self)
        if tp is CumCount:
            return dtm.IDX_DTYPE
        if tp in {CumMin, CumMax}:
            return node.input[0]._resolve_dtype(schema)

        # map_dtype(cum::dtypes::cum_prod)
        # map_dtype(cum::dtypes::cum_sum)
        return super()._resolve_dtype(schema, node)


class RollingWindow(Function, options=FunctionOptions.length_preserving):
    __slots__ = ("options",)
    options: RollingOptionsFixedWindow

    def to_function_expr(self, *inputs: ExprIR) -> RollingExpr[Self]:
        from narwhals._plan.expressions.expr import RollingExpr

        options = self.function_options
        return RollingExpr(input=inputs, function=self, options=options)

    def _resolve_dtype(self, schema: FrozenSchema, node: FunctionExpr[Function]) -> DType:
        return {
            RollingSum: dtm.sum_dtype,
            RollingMean: dtm.moment_dtype,
            RollingVar: dtm.var_dtype,
            RollingStd: dtm.moment_dtype,
        }[type(self)](node.input[0]._resolve_dtype(schema))


class NullCount(Function, options=FunctionOptions.aggregation):
    def _resolve_dtype(self, schema: FrozenSchema, node: FunctionExpr[Function]) -> DType:
        return dtm.IDX_DTYPE


class Exp(Function, options=FunctionOptions.elementwise):
    def _resolve_dtype(self, schema: FrozenSchema, node: FunctionExpr[Function]) -> DType:
        return dtm.float_dtype(node.input[0]._resolve_dtype(schema))


class Diff(Function, options=FunctionOptions.length_preserving):
    def _resolve_dtype(self, schema: FrozenSchema, node: FunctionExpr[Function]) -> DType:
        return dtm.diff_dtype(node.input[0]._resolve_dtype(schema))


# fmt: off
class Abs(_SameDType, options=FunctionOptions.elementwise): ...
class Sqrt(_NumericToFloatDType, options=FunctionOptions.elementwise): ...
class Ceil(_SameDType, options=FunctionOptions.elementwise): ...
class Floor(_SameDType, options=FunctionOptions.elementwise): ...
class DropNulls(_SameDType, options=FunctionOptions.row_separable): ...
class ModeAll(_SameDType): ...
class ModeAny(_SameDType, options=FunctionOptions.aggregation): ...
class Kurtosis(_F64DType, options=FunctionOptions.aggregation): ...
class Skew(_F64DType, options=FunctionOptions.aggregation): ...
class Clip(_SameDType, options=FunctionOptions.elementwise):
    def unwrap_input(self, node: FunctionExpr[Self], /) -> tuple[ExprIR, ExprIR, ExprIR]:
        expr, lower_bound, upper_bound = node.input
        return expr, lower_bound, upper_bound
class ClipLower(_SameDType, options=FunctionOptions.elementwise):
    def unwrap_input(self, node: FunctionExpr[Self], /) -> tuple[ExprIR, ExprIR]:
        expr, lower_bound = node.input
        return expr, lower_bound
class ClipUpper(_SameDType, options=FunctionOptions.elementwise):
    def unwrap_input(self, node: FunctionExpr[Self], /) -> tuple[ExprIR, ExprIR]:
        expr, upper_bound = node.input
        return expr, upper_bound
class CumCount(CumAgg): ...
class CumMin(CumAgg): ...
class CumMax(CumAgg): ...
class CumProd(CumAgg): ...
class CumSum(CumAgg): ...
class RollingSum(RollingWindow): ...
class RollingMean(RollingWindow): ...
class RollingVar(RollingWindow): ...
class RollingStd(RollingWindow): ...
class Unique(_SameDType): ...
class SumHorizontal(HorizontalFunction): ... # map_to_supertype + https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L404-L409
class MinHorizontal(HorizontalFunction): ... # map_to_supertype
class MaxHorizontal(HorizontalFunction): ... # map_to_supertype
class MeanHorizontal(HorizontalFunction): ... # map_to_supertype + https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L410-L420
class Coalesce(HorizontalFunction): ... # map_to_supertype
# fmt: on
class Hist(Function):
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

    def _resolve_dtype(self, schema: FrozenSchema, node: FunctionExpr[Function]) -> DType:
        # NOTE: Need to adapt for `include_category`
        # https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L220-L243
        if self.include_breakpoint:
            return dtm.Struct({"breakpoint": dtm.F64, "count": dtm.IDX_DTYPE})
        return dtm.IDX_DTYPE


class HistBins(Hist):
    __slots__ = ("bins",)
    bins: Seq[float]


class HistBinCount(Hist):
    __slots__ = ("bin_count",)
    bin_count: int


class Log(Function, options=FunctionOptions.elementwise):
    __slots__ = ("base",)
    base: float

    def _resolve_dtype(self, schema: FrozenSchema, node: FunctionExpr[Function]) -> DType:
        dtype = node.input[0]._resolve_dtype(schema)
        return dtype if dtype.is_float() else dtm.F64


class Pow(Function, options=FunctionOptions.elementwise):
    """N-ary (base, exponent)."""

    def unwrap_input(self, node: FunctionExpr[Self], /) -> tuple[ExprIR, ExprIR]:
        base, exponent = node.input
        return base, exponent

    def _resolve_dtype(self, schema: FrozenSchema, node: FunctionExpr[Function]) -> DType:
        base = node.input[0]._resolve_dtype(schema)
        if base.is_integer() and (exp := node.input[1]._resolve_dtype(schema)).is_float():
            return exp
        return base


class FillNull(Function, options=FunctionOptions.elementwise):  # map_to_supertype
    """N-ary (expr, value)."""

    def unwrap_input(self, node: FunctionExpr[Self], /) -> tuple[ExprIR, ExprIR]:
        expr, value = node.input
        return expr, value


class FillNan(_SameDType, options=FunctionOptions.elementwise):
    """N-ary (expr, value)."""

    def unwrap_input(self, node: FunctionExpr[Self], /) -> tuple[ExprIR, ExprIR]:
        expr, value = node.input
        return expr, value


class FillNullWithStrategy(_SameDType):
    __slots__ = ("limit", "strategy")
    strategy: FillNullStrategy
    limit: int | None


class Shift(_SameDType, options=FunctionOptions.length_preserving):
    __slots__ = ("n",)
    n: int


class Rank(Function):
    __slots__ = ("options",)
    options: RankOptions

    def _resolve_dtype(self, schema: FrozenSchema, node: FunctionExpr[Function]) -> DType:
        return dtm.F64 if self.options.method == "average" else dtm.IDX_DTYPE


class Round(_SameDType, options=FunctionOptions.elementwise):
    __slots__ = ("decimals",)
    decimals: int


class EwmMean(_NumericToFloatDType, options=FunctionOptions.length_preserving):
    __slots__ = ("options",)
    options: EWMOptions


class ReplaceStrict(Function, options=FunctionOptions.elementwise):
    __slots__ = ("new", "old", "return_dtype")
    old: Seq[Any]
    new: Seq[Any]
    return_dtype: DType | None

    def _resolve_dtype(self, schema: FrozenSchema, node: FunctionExpr[Function]) -> DType:
        if dtype := self.return_dtype:
            return dtype
        # NOTE: polars would use the dtype of `new` here
        # Would need to run a sample through `common.py_to_narwhals_dtype`
        # https://github.com/narwhals-dev/narwhals/blob/a44792702c107c97b97e3f4a42977c0c9213e1d4/narwhals/_plan/common.py#L52-L67
        # https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L773
        # https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L777
        # https://github.com/pola-rs/polars/blob/675f5b312adfa55b071467d963f8f4a23842fc1e/crates/polars-plan/src/plans/aexpr/function_expr/schema.rs#L781
        return super()._resolve_dtype(schema, node)


# NOTE: similar to `ReplaceStrict._resolve_dtype`, but use `get_supertype(new, default)`
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
