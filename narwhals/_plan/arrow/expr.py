from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import pyarrow as pa  # ignore-banned-import
import pyarrow.compute as pc  # ignore-banned-import

from narwhals._arrow.utils import chunked_array, narwhals_to_native_dtype
from narwhals._plan.arrow.series import ArrowSeries
from narwhals._plan.common import into_dtype
from narwhals._plan.literal import is_literal_scalar
from narwhals._plan.protocols import Dispatch, EagerExpr, EagerScalar
from narwhals._utils import Version, _StoresNative
from narwhals.exceptions import InvalidOperationError, ShapeError

if TYPE_CHECKING:
    from typing_extensions import Self, TypeAlias

    from narwhals._arrow.typing import ChunkedArrayAny, Incomplete
    from narwhals._plan import expr
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


class ArrowExpr(
    Dispatch["ArrowDataFrame", "ArrowExpr | ArrowScalar"],
    _StoresNative["ChunkedArrayAny"],
    EagerExpr["ArrowDataFrame", ArrowSeries],
):
    _evaluated: ArrowSeries

    @property
    def name(self) -> str:
        return self._evaluated.name

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

    @classmethod
    def from_ir(
        cls, value: expr.Literal[DummySeries[ChunkedArrayAny]], name: str = "", /
    ) -> Self:
        nw_ser = value.unwrap()
        return cls.from_native(nw_ser.to_native(), name or value.name, nw_ser.version)

    def col(self, node: expr.Column, frame: ArrowDataFrame, name: str) -> Self:
        return self.from_native(frame.native.column(node.name), name)

    def lit(
        self,
        node: expr.Literal[NonNestedLiteral] | expr.Literal[DummySeries[ChunkedArrayAny]],
        name: str,
    ) -> ArrowScalar | Self:
        if is_literal_scalar(node):
            return ArrowScalar.from_ir(node, name)
        return self.from_ir(node, name)

    @overload
    def _with_native(self, result: ChunkedArrayAny, name: str = ..., /) -> Self: ...
    @overload
    def _with_native(self, result: NativeScalar, name: str = ..., /) -> ArrowScalar: ...
    @overload
    def _with_native(
        self, result: ChunkedArrayAny | NativeScalar, name: str = ..., /
    ) -> ArrowScalar | Self: ...
    def _with_native(
        self, result: ChunkedArrayAny | NativeScalar, name: str = "", /
    ) -> ArrowScalar | Self:
        if isinstance(result, pa.Scalar):
            return ArrowScalar.from_native(result, name, version=self.version)
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

    def cast(  # type: ignore[override]
        self, node: expr.Cast, frame: ArrowDataFrame, name: str
    ) -> ArrowScalar | Self:
        data_type = narwhals_to_native_dtype(node.dtype, frame.version)
        native = self._dispatch(node.expr, frame, name).native
        return self._with_native(pc.cast(native, data_type), name)

    def sort(self, node: expr.Sort, frame: ArrowDataFrame, name: str) -> ArrowExpr:
        raise NotImplementedError

    def sort_by(self, node: expr.SortBy, frame: ArrowDataFrame, name: str) -> ArrowExpr:
        raise NotImplementedError

    def filter(self, node: expr.Filter, frame: ArrowDataFrame, name: str) -> ArrowExpr:
        raise NotImplementedError

    def first(self, node: First, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        native = self._dispatch(node.expr, frame, name).to_series().native
        result: NativeScalar = (
            native[0] if (len(native)) else pa.scalar(None, native.type)
        )
        return self._with_native(result, name)

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
        result: NativeScalar = pc.max(
            self._dispatch(node.expr, frame, name).to_series().native
        )
        return self._with_native(result, name)

    def mean(self, node: Mean, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        raise NotImplementedError

    def median(self, node: Median, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        raise NotImplementedError

    def min(self, node: Min, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        raise NotImplementedError


class ArrowScalar(
    Dispatch["ArrowDataFrame", "ArrowScalar"],
    _StoresNative[NativeScalar],
    EagerScalar["ArrowDataFrame", ArrowSeries],
):
    _name: str
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

    @classmethod
    def from_ir(cls, value: expr.Literal[NonNestedLiteral], name: str, /) -> Self:
        return cls.from_python(value.unwrap(), name, dtype=value.dtype)

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

    def cast(self, node: expr.Cast, frame: ArrowDataFrame, name: str) -> ArrowScalar:
        data_type = narwhals_to_native_dtype(node.dtype, frame.version)
        native = self._dispatch(node.expr, frame, name).native
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
        native = self._dispatch(node.expr, frame, name).native
        return self._with_native(pa.scalar(1 if native.is_valid else 0), name)
