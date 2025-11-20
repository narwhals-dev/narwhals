from __future__ import annotations

# ruff: noqa: PLC0414
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

from narwhals._typing_compat import TypeVar
from narwhals._utils import _StoresNative as StoresNative

if TYPE_CHECKING:
    import pyarrow as pa
    import pyarrow.compute as pc
    from pyarrow.lib import (
        Date32Type,
        Int8Type,
        Int16Type,
        Int32Type,
        Int64Type,
        LargeStringType as LargeStringType,
        StringType as StringType,
        Uint8Type,
        Uint16Type,
        Uint32Type,
        Uint64Type,
    )
    from typing_extensions import ParamSpec, TypeAlias

    from narwhals._native import NativeDataFrame, NativeSeries
    from narwhals.typing import SizedMultiIndexSelector as _SizedMultiIndexSelector

    StringScalar: TypeAlias = "Scalar[StringType | LargeStringType]"
    IntegerType: TypeAlias = "Int8Type | Int16Type | Int32Type | Int64Type | Uint8Type | Uint16Type | Uint32Type | Uint64Type"
    IntegerScalar: TypeAlias = "Scalar[IntegerType]"
    DateScalar: TypeAlias = "Scalar[Date32Type]"

    class NativeArrowSeries(NativeSeries, Protocol):
        @property
        def chunks(self) -> list[Any]: ...

    class NativeArrowDataFrame(NativeDataFrame, Protocol):
        def column(self, *args: Any, **kwds: Any) -> NativeArrowSeries: ...
        @property
        def columns(self) -> Sequence[NativeArrowSeries]: ...

    P = ParamSpec("P")

    class VectorFunction(Protocol[P]):
        def __call__(
            self, native: ChunkedArrayAny, *args: P.args, **kwds: P.kwargs
        ) -> ChunkedArrayAny: ...


ScalarT = TypeVar("ScalarT", bound="pa.Scalar[Any]", default="pa.Scalar[Any]")
ScalarPT_contra = TypeVar(
    "ScalarPT_contra",
    bound="pa.Scalar[Any]",
    default="pa.Scalar[Any]",
    contravariant=True,
)
ScalarRT_co = TypeVar(
    "ScalarRT_co", bound="pa.Scalar[Any]", default="pa.Scalar[Any]", covariant=True
)
NumericOrTemporalScalar: TypeAlias = "pc.NumericOrTemporalScalar"
NumericOrTemporalScalarT = TypeVar(
    "NumericOrTemporalScalarT",
    bound=NumericOrTemporalScalar,
    default=NumericOrTemporalScalar,
)


class UnaryFunction(Protocol[ScalarPT_contra, ScalarRT_co]):
    @overload
    def __call__(self, data: ScalarPT_contra, *args: Any, **kwds: Any) -> ScalarRT_co: ...

    @overload
    def __call__(
        self, data: ChunkedArray[ScalarPT_contra], *args: Any, **kwds: Any
    ) -> ChunkedArray[ScalarRT_co]: ...

    @overload
    def __call__(
        self, data: ChunkedOrScalar[ScalarPT_contra], *args: Any, **kwds: Any
    ) -> ChunkedOrScalar[ScalarRT_co]: ...
    @overload
    def __call__(
        self, data: Array[ScalarPT_contra], *args: Any, **kwds: Any
    ) -> Array[ScalarRT_co]: ...
    @overload
    def __call__(
        self, data: ChunkedOrArray[ScalarPT_contra], *args: Any, **kwds: Any
    ) -> ChunkedOrArray[ScalarRT_co]: ...

    def __call__(
        self, data: Arrow[ScalarPT_contra], *args: Any, **kwds: Any
    ) -> Arrow[ScalarRT_co]: ...


class BinaryFunction(Protocol[ScalarPT_contra, ScalarRT_co]):
    @overload
    def __call__(self, x: ScalarPT_contra, y: ScalarPT_contra, /) -> ScalarRT_co: ...

    @overload
    def __call__(
        self, x: ChunkedArray[ScalarPT_contra], y: ChunkedArray[ScalarPT_contra], /
    ) -> ChunkedArray[ScalarRT_co]: ...

    @overload
    def __call__(
        self, x: ScalarPT_contra, y: ChunkedArray[ScalarPT_contra], /
    ) -> ChunkedArray[ScalarRT_co]: ...

    @overload
    def __call__(
        self, x: ChunkedArray[ScalarPT_contra], y: ScalarPT_contra, /
    ) -> ChunkedArray[ScalarRT_co]: ...

    @overload
    def __call__(
        self, x: ChunkedOrScalar[ScalarPT_contra], y: ChunkedOrScalar[ScalarPT_contra], /
    ) -> ChunkedOrScalar[ScalarRT_co]: ...

    def __call__(
        self, x: ChunkedOrScalar[ScalarPT_contra], y: ChunkedOrScalar[ScalarPT_contra], /
    ) -> ChunkedOrScalar[ScalarRT_co]: ...


class BinaryComp(
    BinaryFunction[ScalarPT_contra, "pa.BooleanScalar"], Protocol[ScalarPT_contra]
): ...


class BinaryLogical(BinaryFunction["pa.BooleanScalar", "pa.BooleanScalar"], Protocol): ...


BinaryNumericTemporal: TypeAlias = BinaryFunction[
    NumericOrTemporalScalarT, NumericOrTemporalScalarT
]
DataType: TypeAlias = "pa.DataType"
DataTypeT = TypeVar("DataTypeT", bound=DataType, default=Any)
DataTypeT_co = TypeVar("DataTypeT_co", bound=DataType, covariant=True, default=Any)
ScalarT_co = TypeVar("ScalarT_co", bound="pa.Scalar[Any]", covariant=True, default=Any)
Scalar: TypeAlias = "pa.Scalar[DataTypeT_co]"
Array: TypeAlias = "pa.Array[ScalarT_co]"
ChunkedArray: TypeAlias = "pa.ChunkedArray[ScalarT_co]"
ChunkedOrScalar: TypeAlias = "ChunkedArray[ScalarT_co] | ScalarT_co"
ChunkedOrArray: TypeAlias = "ChunkedArray[ScalarT_co] | Array[ScalarT_co]"
ScalarAny: TypeAlias = "Scalar[Any]"
ArrayAny: TypeAlias = "Array[Any]"
ChunkedArrayAny: TypeAlias = "ChunkedArray[Any]"
ChunkedOrScalarAny: TypeAlias = "ChunkedOrScalar[ScalarAny]"
ChunkedOrArrayAny: TypeAlias = "ChunkedOrArray[ScalarAny]"
ChunkedOrArrayT = TypeVar("ChunkedOrArrayT", ChunkedArrayAny, ArrayAny)
Indices: TypeAlias = "_SizedMultiIndexSelector[ChunkedOrArray[pc.IntegerScalar]]"

Arrow: TypeAlias = "ChunkedOrScalar[ScalarT_co] | Array[ScalarT_co]"
ArrowAny: TypeAlias = "ChunkedOrScalarAny | ArrayAny"
NativeScalar: TypeAlias = ScalarAny
BinOp: TypeAlias = Callable[..., ChunkedOrScalarAny]
StoresNativeT_co = TypeVar(
    "StoresNativeT_co", bound=StoresNative[ChunkedOrScalarAny], covariant=True
)
DataTypeRemap: TypeAlias = Mapping[DataType, DataType]
NullPlacement: TypeAlias = Literal["at_start", "at_end"]

JoinTypeSubset: TypeAlias = Literal[
    "inner", "left outer", "full outer", "left anti", "left semi"
]
"""Only the `pyarrow` `JoinType`'s we use in narwhals"""

RankMethodSingle: TypeAlias = Literal["min", "max", "dense", "ordinal"]
"""Subset of `narwhals` `RankMethod` that is supported directly in `pyarrow`.

`"average"` requires calculating both `"min"` and `"max"`.
"""
