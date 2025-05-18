"""Mock version of current narwhals API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._plan import aggregation as agg
from narwhals._plan import boolean
from narwhals._plan import expr
from narwhals._plan import operators as ops
from narwhals.dtypes import DType
from narwhals.dtypes import Unknown

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan.common import ExprIR
    from narwhals.typing import NativeSeries


# NOTE: Overly simplified placeholders for mocking typing
# Entirely ignoring namespace + function binding
class DummyExpr:
    _ir: ExprIR

    def __repr__(self) -> str:
        return f"Narwhals DummyExpr:\n{self._ir!r}"

    @classmethod
    def _from_ir(cls, ir: ExprIR, /) -> Self:
        obj = cls.__new__(cls)
        obj._ir = ir
        return obj

    def alias(self, name: str) -> Self:
        return self._from_ir(expr.Alias(expr=self._ir, name=name))

    def cast(self, dtype: DType | type[DType]) -> Self:
        dtype = dtype if isinstance(dtype, DType) else Unknown()
        return self._from_ir(expr.Cast(expr=self._ir, dtype=dtype))

    def count(self) -> Self:
        return self._from_ir(agg.Count(expr=self._ir))

    def max(self) -> Self:
        return self._from_ir(agg.Max(expr=self._ir))

    def mean(self) -> Self:
        return self._from_ir(agg.Mean(expr=self._ir))

    def min(self) -> Self:
        return self._from_ir(agg.Min(expr=self._ir))

    def median(self) -> Self:
        return self._from_ir(agg.Median(expr=self._ir))

    def n_unique(self) -> Self:
        return self._from_ir(agg.NUnique(expr=self._ir))

    def sum(self) -> Self:
        return self._from_ir(agg.Sum(expr=self._ir))

    def __eq__(self, other: DummyExpr) -> DummyExpr:  # type: ignore[override]
        op = ops.Eq()
        return op.to_binary_expr(self._ir, other._ir).to_narwhals()

    def __ne__(self, other: DummyExpr) -> DummyExpr:  # type: ignore[override]
        op = ops.NotEq()
        return op.to_binary_expr(self._ir, other._ir).to_narwhals()

    def __lt__(self, other: DummyExpr) -> DummyExpr:
        op = ops.Lt()
        return op.to_binary_expr(self._ir, other._ir).to_narwhals()

    def __le__(self, other: DummyExpr) -> DummyExpr:
        op = ops.LtEq()
        return op.to_binary_expr(self._ir, other._ir).to_narwhals()

    def __gt__(self, other: DummyExpr) -> DummyExpr:
        op = ops.Gt()
        return op.to_binary_expr(self._ir, other._ir).to_narwhals()

    def __ge__(self, other: DummyExpr) -> DummyExpr:
        op = ops.GtEq()
        return op.to_binary_expr(self._ir, other._ir).to_narwhals()

    def __add__(self, other: DummyExpr) -> DummyExpr:
        op = ops.Add()
        return op.to_binary_expr(self._ir, other._ir).to_narwhals()

    def __sub__(self, other: DummyExpr) -> DummyExpr:
        op = ops.Sub()
        return op.to_binary_expr(self._ir, other._ir).to_narwhals()

    def __mul__(self, other: DummyExpr) -> DummyExpr:
        op = ops.Multiply()
        return op.to_binary_expr(self._ir, other._ir).to_narwhals()

    def __truediv__(self, other: DummyExpr) -> DummyExpr:
        op = ops.TrueDivide()
        return op.to_binary_expr(self._ir, other._ir).to_narwhals()

    def __floordiv__(self, other: DummyExpr) -> DummyExpr:
        op = ops.FloorDivide()
        return op.to_binary_expr(self._ir, other._ir).to_narwhals()

    def __mod__(self, other: DummyExpr) -> DummyExpr:
        op = ops.Modulus()
        return op.to_binary_expr(self._ir, other._ir).to_narwhals()

    def __and__(self, other: DummyExpr) -> DummyExpr:
        op = ops.And()
        return op.to_binary_expr(self._ir, other._ir).to_narwhals()

    def __or__(self, other: DummyExpr) -> DummyExpr:
        op = ops.Or()
        return op.to_binary_expr(self._ir, other._ir).to_narwhals()

    def __invert__(self) -> DummyExpr:
        return boolean.Not().to_function_expr().to_narwhals()


class DummyCompliantExpr:
    _ir: ExprIR

    @classmethod
    def _from_ir(cls, ir: ExprIR, /) -> Self:
        obj = cls.__new__(cls)
        obj._ir = ir
        return obj


class DummySeries:
    _compliant: DummyCompliantSeries

    @property
    def dtype(self) -> DType:
        return self._compliant.dtype

    @property
    def name(self) -> str:
        return self._compliant.name

    @classmethod
    def from_native(cls, native: NativeSeries, /) -> Self:
        obj = cls.__new__(cls)
        obj._compliant = DummyCompliantSeries.from_native(native)
        return obj


class DummyCompliantSeries:
    _native: NativeSeries
    _name: str

    @property
    def dtype(self) -> DType:
        from narwhals.dtypes import Float64

        return Float64()

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def from_native(cls, native: NativeSeries, /) -> Self:
        from narwhals.utils import _hasattr_static

        name: str = "<PLACEHOLDER>"

        if _hasattr_static(native, "name"):
            name = getattr(native, "name", name)
        obj = cls.__new__(cls)
        obj._native = native
        obj._name = name
        return obj
