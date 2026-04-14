from __future__ import annotations

# NOTE: Needed to avoid naming collisions
# - Any
import typing as t
from typing import TYPE_CHECKING, ClassVar, Generic

import narwhals._plan.dtypes_mapper as dtm
from narwhals._plan import _parameters as params
from narwhals._plan._dispatch import DispatcherOptions
from narwhals._plan._dtype import ResolveDType
from narwhals._plan._flags import FunctionFlags
from narwhals._plan._function import Function, HorizontalFunction
from narwhals._plan.typing import NativeSeriesT, NativeSeriesT_co

if TYPE_CHECKING:
    from narwhals._plan._expr_ir import ExprIR
    from narwhals._plan.expressions import FunctionExpr as FExpr
    from narwhals._plan.expressions.literal import LitSeries
    from narwhals._plan.series import Series
    from narwhals._plan.typing import Seq
    from narwhals.typing import ClosedInterval

# NOTE: See https://github.com/astral-sh/ty/issues/1777#issuecomment-3618906859
AGGREGATION = FunctionFlags.AGGREGATION
ELEMENTWISE = FunctionFlags.ELEMENTWISE
LENGTH_PRESERVING = FunctionFlags.LENGTH_PRESERVING
map_first = ResolveDType.function.map_first
renamed = DispatcherOptions.renamed


# fmt: off
class BooleanFunction(Function, flags=ELEMENTWISE, dtype=dtm.BOOL):...
class _HorizontalBoolean(HorizontalFunction, BooleanFunction, dtype=dtm.BOOL):
    __slots__ = ("ignore_nulls",)
    ignore_nulls: bool
class All(BooleanFunction, flags=AGGREGATION): ...
class Any(BooleanFunction, flags=AGGREGATION): ...
class AllHorizontal(_HorizontalBoolean):...
class AnyHorizontal(_HorizontalBoolean):...
class IsDuplicated(BooleanFunction, flags=LENGTH_PRESERVING): ...
class IsFinite(BooleanFunction): ...
class IsFirstDistinct(BooleanFunction, flags=LENGTH_PRESERVING): ...
class IsLastDistinct(BooleanFunction, flags=LENGTH_PRESERVING): ...
class IsNan(BooleanFunction): ...
class IsNull(BooleanFunction): ...
class IsNotNan(BooleanFunction): ...
class IsNotNull(BooleanFunction): ...
class IsUnique(BooleanFunction, flags=LENGTH_PRESERVING): ...
class Not(BooleanFunction, dispatch=renamed("not_"), dtype=map_first(lambda dtype: dtype if dtype.is_integer() else dtm.BOOL)): ...
# fmt: on
class IsBetween(BooleanFunction):
    __slots__ = ("closed",)
    closed: ClosedInterval
    __function_parameters__: ClassVar[params.Ternary] = params.Ternary()


class IsInSeq(BooleanFunction):
    __slots__ = ("other",)
    other: Seq[t.Any]

    def __repr__(self) -> str:
        return "is_in"

    @classmethod
    def from_iterable(cls, other: t.Iterable[t.Any], /) -> IsInSeq:
        if not isinstance(other, (str, bytes)):
            return IsInSeq(other=tuple(other))
        msg = f"`is_in` doesn't accept `str | bytes` as iterables, got: {type(other).__name__}"
        raise TypeError(msg)


class IsInSeries(BooleanFunction, Generic[NativeSeriesT_co]):
    __slots__ = ("other",)
    other: LitSeries[NativeSeriesT_co]

    def __repr__(self) -> str:
        return "is_in"

    @staticmethod
    def from_series(other: Series[NativeSeriesT], /) -> IsInSeries[NativeSeriesT]:
        from narwhals._plan.expressions.literal import lit_series

        return IsInSeries(other=lit_series(other))


class IsInExpr(BooleanFunction):
    # NOTE: *Consider* restricting to non-equal types (https://github.com/pola-rs/polars/pull/22178)
    __function_parameters__: ClassVar[params.Binary] = params.Binary()

    def __repr__(self) -> str:
        return "is_in"


def all_horizontal(*exprs: ExprIR, ignore_nulls: bool = False) -> FExpr[AllHorizontal]:
    return AllHorizontal(ignore_nulls=ignore_nulls).to_function_expr(*exprs)
