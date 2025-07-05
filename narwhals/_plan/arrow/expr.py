from __future__ import annotations

from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._arrow.utils import chunked_array, narwhals_to_native_dtype
from narwhals._plan import expr
from narwhals._plan.arrow.series import ArrowSeries
from narwhals._plan.common import ExprIR, NamedIR, into_dtype
from narwhals._plan.protocols import EagerBroadcast, EagerExpr, EagerScalar
from narwhals._utils import Version
from narwhals.exceptions import InvalidOperationError, ShapeError

if TYPE_CHECKING:
    from typing_extensions import Self, TypeAlias

    from narwhals._arrow.typing import ChunkedArrayAny, Incomplete, ScalarAny
    from narwhals._plan.aggregation import (
        ArgMax,
        ArgMin,
        Count,
        First,
        Last,
        Max,
        Mean,
        Median,
        Min,
        NUnique,
        Quantile,
        Std,
        Sum,
        Var,
    )
    from narwhals._plan.arrow.dataframe import ArrowDataFrame
    from narwhals._plan.dummy import DummySeries
    from narwhals.typing import IntoDType, NonNestedLiteral, PythonLiteral

NativeScalar: TypeAlias = "pa.Scalar[Any]"


class ArrowExpr2(EagerExpr["ArrowDataFrame", ArrowSeries]):
    _evaluated: ArrowSeries

    @property
    def name(self) -> str:
        return self._evaluated.name

    @property
    def version(self) -> Version:
        return self._evaluated.version

    @classmethod
    def from_series(cls, series: ArrowSeries, /) -> Self:
        obj = cls.__new__(cls)
        obj._evaluated = series
        return obj

    @classmethod
    def from_native(
        cls, native: ChunkedArrayAny, name: str = "", /, version: Version = Version.MAIN
    ) -> Self:
        return cls.from_series(ArrowSeries.from_native(native, name, version=version))

    def _with_native(
        self, result: ChunkedArrayAny | NativeScalar, name: str = "", /
    ) -> Self:
        if isinstance(result, pa.Scalar):
            # NOTE: Will need to resolve this eventually
            # Currently the *least bad* option is the single ignore here
            return ArrowScalar.from_native(result, name, version=self.version)  # type: ignore[return-value]
        return super()._with_native(result, name)

    @property
    def native(self) -> ChunkedArrayAny:
        return self._evaluated.native

    def to_series(self) -> ArrowSeries:
        return self._evaluated

    def broadcast(self, length: int, /) -> ArrowSeries:
        if (actual_len := len(self)) != length:
            msg = f"Expected object of length {length}, got {actual_len}."
            raise ShapeError(msg)
        return self._evaluated

    def __len__(self) -> int:
        return len(self._evaluated)

    # NOTE: Dispatch is on `ExprIR`, which is recursive
    # There is only a top-level `NamedIR` per column
    def evaluate(self, named_ir: NamedIR[ExprIR], frame: ArrowDataFrame) -> ArrowExpr2:
        return self._evaluate_inner(named_ir.expr, frame, named_ir.name)

    # NOTE: Don't use `Self`, it breaks the descriptor typing
    # The implementations *can* use `Self`, just not here
    @singledispatchmethod
    def _evaluate_inner(
        self, node: ExprIR, frame: ArrowDataFrame, name: str
    ) -> ArrowExpr2:
        raise NotImplementedError(type(node))

    @_evaluate_inner.register(expr.Cast)
    def cast(self, node: expr.Cast, frame: ArrowDataFrame, name: str) -> Self:
        data_type = narwhals_to_native_dtype(node.dtype, frame.version)
        native = self._evaluate_inner(node.expr, frame, name).native
        return self._with_native(pc.cast(native, data_type), name)

    def sort(self, node: expr.Sort, frame: ArrowDataFrame, name: str) -> ArrowExpr2:
        raise NotImplementedError

    def sort_by(self, node: expr.SortBy, frame: ArrowDataFrame, name: str) -> ArrowExpr2:
        raise NotImplementedError

    def filter(self, node: expr.Filter, frame: ArrowDataFrame, name: str) -> ArrowExpr2:
        raise NotImplementedError

    def first(self, node: First, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        raise NotImplementedError

    def last(self, node: Last, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        raise NotImplementedError

    def arg_min(self, node: ArgMin, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        raise NotImplementedError

    def arg_max(self, node: ArgMax, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        raise NotImplementedError

    def sum(self, node: Sum, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        raise NotImplementedError

    def n_unique(self, node: NUnique, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        raise NotImplementedError

    def std(self, node: Std, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        raise NotImplementedError

    def var(self, node: Var, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        raise NotImplementedError

    def quantile(self, node: Quantile, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        raise NotImplementedError

    def count(self, node: Count, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        raise NotImplementedError

    def max(self, node: Max, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        raise NotImplementedError

    def mean(self, node: Mean, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        raise NotImplementedError

    def median(self, node: Median, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        raise NotImplementedError

    def min(self, node: Min, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        raise NotImplementedError


class ArrowScalar(EagerScalar["ArrowDataFrame", ArrowSeries]):
    _name: str
    _version: Version
    _evaluated: NativeScalar

    @property
    def name(self) -> str:
        return self._name

    @classmethod
    def from_native(
        cls,
        scalar: NativeScalar,
        name: str = "literal",
        /,
        version: Version = Version.MAIN,
    ) -> Self:
        obj = cls.__new__(cls)
        obj._evaluated = scalar
        obj._name = name
        obj._version = version
        return obj

    @classmethod
    def from_python(
        cls,
        value: PythonLiteral,
        name: str = "literal",
        /,
        *,
        dtype: IntoDType | None = None,
        version: Version = Version.MAIN,
    ) -> Self:
        dtype_pa: pa.DataType | None = None
        if dtype:
            dtype = into_dtype(dtype)
            if not isinstance(dtype, version.dtypes.Unknown):
                dtype_pa = narwhals_to_native_dtype(dtype, version)
        # NOTE: PR that fixed this was closed
        # https://github.com/zen-xu/pyarrow-stubs/pull/208
        lit: Incomplete = pa.scalar
        return cls.from_native(lit(value, dtype_pa), name, version)

    @classmethod
    def from_series(cls, series: ArrowSeries) -> Self:
        if len(series) == 1:
            return cls.from_native(series.native[0], series.name, series.version)
        elif len(series) == 0:
            return cls.from_python(
                None, series.name, dtype=series.dtype, version=series.version
            )
        else:
            msg = f"Too long {len(series)!r}"
            raise InvalidOperationError(msg)

    @property
    def native(self) -> NativeScalar:
        return self._evaluated

    def to_series(self) -> ArrowSeries:
        return self.broadcast(1)

    def broadcast(self, length: int) -> ArrowSeries:
        scalar = self.native
        if length == 1:
            chunked = chunked_array([[scalar]])
        else:
            # NOTE: Same issue as `pa.scalar` overlapping overloads
            # https://github.com/zen-xu/pyarrow-stubs/pull/209
            pa_repeat: Incomplete = pa.repeat
            chunked = chunked_array(pa_repeat(scalar, length))
        return ArrowSeries.from_native(chunked, self.name, version=self.version)

    # NOTE: Dispatch is on `ExprIR`, which is recursive
    # There is only a top-level `NamedIR` per column
    def evaluate(self, named_ir: NamedIR[ExprIR], frame: ArrowDataFrame) -> ArrowScalar:
        return self._evaluate_inner(named_ir.expr, frame, named_ir.name)

    @singledispatchmethod
    def _evaluate_inner(
        self, node: ExprIR, frame: ArrowDataFrame, name: str
    ) -> ArrowScalar:
        raise NotImplementedError(type(node))

    @_evaluate_inner.register(expr.Cast)
    def cast(self, node: expr.Cast, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        data_type = narwhals_to_native_dtype(node.dtype, frame.version)
        native = self._evaluate_inner(node.expr, frame, name).native
        return self._with_native(pc.cast(native, data_type), name)

    def filter(self, node: expr.Filter, frame: ArrowDataFrame, name: str) -> Any:
        raise NotImplementedError

    def arg_min(self, node: ArgMin, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        return self._with_native(pa.scalar(0), name)

    def arg_max(self, node: ArgMax, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        return self._with_native(pa.scalar(0), name)

    def n_unique(self, node: NUnique, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        return self._with_native(pa.scalar(1), name)

    def std(self, node: Std, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        return self._with_native(pa.scalar(None, pa.null()), name)

    def var(self, node: Var, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        return self._with_native(pa.scalar(None, pa.null()), name)

    def count(self, node: Count, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        native = self._evaluate_inner(node.expr, frame, name).native
        return self._with_native(pa.scalar(1 if native.is_valid else 0), name)


# NOTE: General expression result
# Mostly elementwise
class ArrowExpr(EagerBroadcast[ArrowSeries]):
    _compliant: ArrowSeries

    @classmethod
    def from_series(cls, series: ArrowSeries) -> Self:
        obj = cls.__new__(cls)
        obj._compliant = series
        return obj

    @classmethod
    def from_native(
        cls,
        native: ChunkedArrayAny,
        name: str = "",
        /,
        *,
        version: Version = Version.MAIN,
    ) -> Self:
        return cls.from_series(ArrowSeries.from_native(native, name, version=version))

    @classmethod
    def from_ir(cls, value: expr.Literal[DummySeries[ChunkedArrayAny]], /) -> Self:
        return cls.from_native(value.unwrap().to_native(), value.name)

    def to_series(self) -> ArrowSeries:
        return self._compliant

    def __len__(self) -> int:
        return len(self._compliant)

    def broadcast(self, length: int, /) -> ArrowSeries:
        if (actual_len := len(self)) != length:
            msg = f"Expected object of length {length}, got {actual_len}."
            raise ShapeError(msg)
        return self._compliant


# NOTE: Aggregation result or scalar
# Should handle broadcasting, without exposing it
class ArrowLiteral(EagerBroadcast[ArrowSeries]):
    _native_scalar: ScalarAny
    _name: str

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return 1

    def broadcast(self, length: int, /) -> ArrowSeries:
        if length == 1:
            chunked = chunked_array([[self._native_scalar]])
        else:
            # NOTE: Same issue as `pa.scalar` overlapping overloads
            # https://github.com/zen-xu/pyarrow-stubs/pull/209
            pa_repeat: Incomplete = pa.repeat
            arr = pa_repeat(self._native_scalar, length)
            chunked = chunked_array(arr)
        return ArrowSeries.from_native(chunked, self.name)

    @classmethod
    def from_series(cls, series: ArrowSeries) -> Self:
        if len(series) == 1:
            return cls.from_scalar(series.native[0], series.name)
        elif len(series) == 0:
            return cls.from_python(None, series.name, dtype=series.dtype)
        else:
            msg = f"Too long {len(series)!r}"
            raise InvalidOperationError(msg)

    def to_series(self) -> ArrowSeries:
        return self.broadcast(1)

    @classmethod
    def from_python(
        cls,
        value: PythonLiteral,
        name: str = "literal",
        /,
        *,
        dtype: IntoDType | None = None,
    ) -> Self:
        version = Version.MAIN
        dtype_pa: pa.DataType | None = None
        if dtype:
            dtype = into_dtype(dtype)
            if not isinstance(dtype, version.dtypes.Unknown):
                dtype_pa = narwhals_to_native_dtype(dtype, version)
        # NOTE: PR that fixed this was closed
        # https://github.com/zen-xu/pyarrow-stubs/pull/208
        lit: Incomplete = pa.scalar
        return cls.from_scalar(lit(value, dtype_pa), name)

    @classmethod
    def from_scalar(cls, scalar: ScalarAny, name: str = "literal", /) -> Self:
        obj = cls.__new__(cls)
        obj._native_scalar = scalar
        obj._name = name
        return obj

    @classmethod
    def from_ir(cls, value: expr.Literal[NonNestedLiteral], /) -> Self:
        return cls.from_python(value.unwrap(), value.name)
