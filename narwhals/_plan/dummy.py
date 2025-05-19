"""Mock version of current narwhals API."""

from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

from narwhals._plan import aggregation as agg
from narwhals._plan import boolean
from narwhals._plan import expr
from narwhals._plan import operators as ops
from narwhals._plan.options import SortMultipleOptions
from narwhals._plan.options import SortOptions
from narwhals._plan.window import Over
from narwhals.dtypes import DType
from narwhals.utils import Version
from narwhals.utils import _hasattr_static
from narwhals.utils import flatten

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._plan.common import ExprIR
    from narwhals._plan.common import Seq
    from narwhals.typing import NativeSeries
    from narwhals.typing import RollingInterpolationMethod


# NOTE: Overly simplified placeholders for mocking typing
# Entirely ignoring namespace + function binding
class DummyExpr:
    _ir: ExprIR
    _version: t.ClassVar[Version] = Version.MAIN

    def __repr__(self) -> str:
        return f"Narwhals DummyExpr ({self.version.name.lower()}):\n{self._ir!r}"

    @classmethod
    def _from_ir(cls, ir: ExprIR, /) -> Self:
        obj = cls.__new__(cls)
        obj._ir = ir
        return obj

    @property
    def version(self) -> Version:
        return self._version

    def alias(self, name: str) -> Self:
        return self._from_ir(expr.Alias(expr=self._ir, name=name))

    def cast(self, dtype: DType | type[DType]) -> Self:
        dtype = dtype if isinstance(dtype, DType) else self.version.dtypes.Unknown()
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

    def first(self) -> Self:
        return self._from_ir(agg.First(expr=self._ir))

    def last(self) -> Self:
        return self._from_ir(agg.Last(expr=self._ir))

    def var(self, *, ddof: int = 1) -> Self:
        return self._from_ir(agg.Var(expr=self._ir, ddof=ddof))

    def std(self, *, ddof: int = 1) -> Self:
        return self._from_ir(agg.Std(expr=self._ir, ddof=ddof))

    def quantile(
        self, quantile: float, interpolation: RollingInterpolationMethod
    ) -> Self:
        return self._from_ir(
            agg.Quantile(expr=self._ir, quantile=quantile, interpolation=interpolation)
        )

    def over(
        self,
        *partition_by: DummyExpr | t.Iterable[DummyExpr],
        order_by: DummyExpr | t.Iterable[DummyExpr] | None = None,
        descending: bool = False,
        nulls_last: bool = False,
    ) -> Self:
        order: tuple[Seq[ExprIR], SortOptions] | None = None
        partition = tuple(expr._ir for expr in flatten(partition_by))
        if not (partition) and order_by is None:
            msg = "At least one of `partition_by` or `order_by` must be specified."
            raise TypeError(msg)
        if order_by is not None:
            by = tuple(expr._ir for expr in flatten([order_by]))
            options = SortOptions(descending=descending, nulls_last=nulls_last)
            order = by, options
        return self._from_ir(Over().to_window_expr(self._ir, partition, order))

    def sort_by(
        self,
        by: DummyExpr | t.Iterable[DummyExpr],
        *more_by: DummyExpr,
        descending: bool | t.Iterable[bool] = False,
        nulls_last: bool | t.Iterable[bool] = False,
    ) -> Self:
        if more_by:
            by = (by, *more_by) if isinstance(by, DummyExpr) else (*by, *more_by)
        else:
            by = (by,) if isinstance(by, DummyExpr) else tuple(by)
        sort_by = tuple(key._ir for key in by)
        desc = (descending,) if isinstance(descending, bool) else tuple(descending)
        nulls = (nulls_last,) if isinstance(nulls_last, bool) else tuple(nulls_last)
        options = SortMultipleOptions(descending=desc, nulls_last=nulls)
        return self._from_ir(expr.SortBy(expr=self._ir, by=sort_by, options=options))

    def __eq__(self, other: DummyExpr) -> Self:  # type: ignore[override]
        op = ops.Eq()
        return self._from_ir(op.to_binary_expr(self._ir, other._ir))

    def __ne__(self, other: DummyExpr) -> Self:  # type: ignore[override]
        op = ops.NotEq()
        return self._from_ir(op.to_binary_expr(self._ir, other._ir))

    def __lt__(self, other: DummyExpr) -> Self:
        op = ops.Lt()
        return self._from_ir(op.to_binary_expr(self._ir, other._ir))

    def __le__(self, other: DummyExpr) -> Self:
        op = ops.LtEq()
        return self._from_ir(op.to_binary_expr(self._ir, other._ir))

    def __gt__(self, other: DummyExpr) -> Self:
        op = ops.Gt()
        return self._from_ir(op.to_binary_expr(self._ir, other._ir))

    def __ge__(self, other: DummyExpr) -> Self:
        op = ops.GtEq()
        return self._from_ir(op.to_binary_expr(self._ir, other._ir))

    def __add__(self, other: DummyExpr) -> Self:
        op = ops.Add()
        return self._from_ir(op.to_binary_expr(self._ir, other._ir))

    def __sub__(self, other: DummyExpr) -> Self:
        op = ops.Sub()
        return self._from_ir(op.to_binary_expr(self._ir, other._ir))

    def __mul__(self, other: DummyExpr) -> Self:
        op = ops.Multiply()
        return self._from_ir(op.to_binary_expr(self._ir, other._ir))

    def __truediv__(self, other: DummyExpr) -> Self:
        op = ops.TrueDivide()
        return self._from_ir(op.to_binary_expr(self._ir, other._ir))

    def __floordiv__(self, other: DummyExpr) -> Self:
        op = ops.FloorDivide()
        return self._from_ir(op.to_binary_expr(self._ir, other._ir))

    def __mod__(self, other: DummyExpr) -> Self:
        op = ops.Modulus()
        return self._from_ir(op.to_binary_expr(self._ir, other._ir))

    def __and__(self, other: DummyExpr) -> Self:
        op = ops.And()
        return self._from_ir(op.to_binary_expr(self._ir, other._ir))

    def __or__(self, other: DummyExpr) -> Self:
        op = ops.Or()
        return self._from_ir(op.to_binary_expr(self._ir, other._ir))

    def __invert__(self) -> Self:
        return self._from_ir(boolean.Not().to_function_expr(self._ir))


class DummyExprV1(DummyExpr):
    _version: t.ClassVar[Version] = Version.V1


class DummyCompliantExpr:
    _ir: ExprIR
    _version: Version

    @property
    def version(self) -> Version:
        return self._version

    @classmethod
    def _from_ir(cls, ir: ExprIR, /, version: Version) -> Self:
        obj = cls.__new__(cls)
        obj._ir = ir
        obj._version = version
        return obj

    def to_narwhals(self) -> DummyExpr:
        if self.version is Version.MAIN:
            return DummyExpr._from_ir(self._ir)
        return DummyExprV1._from_ir(self._ir)


class DummySeries:
    _compliant: DummyCompliantSeries
    _version: t.ClassVar[Version] = Version.MAIN

    @property
    def version(self) -> Version:
        return self._version

    @property
    def dtype(self) -> DType:
        return self._compliant.dtype

    @property
    def name(self) -> str:
        return self._compliant.name

    @classmethod
    def from_native(cls, native: NativeSeries, /) -> Self:
        obj = cls.__new__(cls)
        obj._compliant = DummyCompliantSeries.from_native(native, cls._version)
        return obj


class DummySeriesV1(DummySeries):
    _version: t.ClassVar[Version] = Version.V1


class DummyCompliantSeries:
    _native: NativeSeries
    _name: str
    _version: Version

    @property
    def version(self) -> Version:
        return self._version

    @property
    def dtype(self) -> DType:
        return self.version.dtypes.Float64()

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def from_native(cls, native: NativeSeries, /, version: Version) -> Self:
        name: str = "<PLACEHOLDER>"
        if _hasattr_static(native, "name"):
            name = getattr(native, "name", name)
        obj = cls.__new__(cls)
        obj._native = native
        obj._name = name
        obj._version = version
        return obj
