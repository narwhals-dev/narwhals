from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from narwhals._arrow.utils import chunked_array, narwhals_to_native_dtype
from narwhals._plan.arrow.series import ArrowSeries
from narwhals._plan.common import into_dtype
from narwhals._plan.protocols import SupportsBroadcast
from narwhals._utils import Version
from narwhals.exceptions import InvalidOperationError, ShapeError

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.typing import ChunkedArrayAny, Incomplete, ScalarAny
    from narwhals._plan import expr
    from narwhals._plan.dummy import DummySeries
    from narwhals.typing import IntoDType, NonNestedLiteral, PythonLiteral


# NOTE: General expression result
# Mostly elementwise
class ArrowExpr(SupportsBroadcast[ArrowSeries]):
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
class ArrowLiteral(SupportsBroadcast[ArrowSeries]):
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
