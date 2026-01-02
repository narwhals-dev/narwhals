from __future__ import annotations

# ruff: noqa: PLC0414
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

from narwhals._typing_compat import TypeVar
from narwhals._utils import _StoresNative as StoresNative

if TYPE_CHECKING:
    import pyarrow as pa
    import pyarrow.compute as pc
    from pyarrow import lib, types
    from pyarrow.lib import (
        BoolType as BoolType,
        Date32Type,
        Int8Type,
        Int16Type,
        Int32Type,
        Int64Type,
        LargeStringType as _LargeStringType,
        StringType as _StringType,
        Uint8Type,
        Uint16Type,
        Uint32Type,
        Uint64Type,
    )
    from typing_extensions import ParamSpec, TypeAlias

    from narwhals._native import NativeDataFrame, NativeSeries
    from narwhals.typing import SizedMultiIndexSelector as _SizedMultiIndexSelector

    UInt32Type: TypeAlias = "Uint32Type"
    StringType: TypeAlias = "_StringType | _LargeStringType"
    IntegerType: TypeAlias = "Int8Type | Int16Type | Int32Type | Int64Type | Uint8Type | Uint16Type | Uint32Type | Uint64Type"
    StringScalar: TypeAlias = "Scalar[StringType]"
    IntegerScalar: TypeAlias = "Scalar[IntegerType]"
    DateScalar: TypeAlias = "Scalar[Date32Type]"
    ListScalar: TypeAlias = "Scalar[pa.ListType[DataTypeT_co]]"
    BooleanScalar: TypeAlias = "Scalar[BoolType]"
    """Only use this for a parameter type, not as a return type!"""
    NumericScalar: TypeAlias = "pc.NumericScalar"

    PrimitiveNumericType: TypeAlias = "types._Integer | types._Floating"
    NumericType: TypeAlias = "PrimitiveNumericType | types._Decimal"
    NumericOrTemporalType: TypeAlias = "NumericType | types._Temporal"
    StringOrBinaryType: TypeAlias = "StringType | lib.StringViewType | lib.BinaryType | lib.LargeBinaryType | lib.BinaryViewType"
    BasicType: TypeAlias = (
        "NumericOrTemporalType | StringOrBinaryType | BoolType | lib.NullType"
    )
    NonListNestedType: TypeAlias = "pa.StructType | pa.DictionaryType[Any, Any] | pa.MapType[Any, Any] | pa.UnionType"
    NonListType: TypeAlias = "BasicType | NonListNestedType"
    NestedType: TypeAlias = "NonListNestedType | pa.ListType[Any]"
    NonListTypeT = TypeVar("NonListTypeT", bound="NonListType")
    ListTypeT = TypeVar("ListTypeT", bound="pa.ListType[Any]")

    class NativeArrowSeries(NativeSeries, Protocol):
        @property
        def chunks(self) -> list[Any]: ...

    class NativeArrowDataFrame(NativeDataFrame, Protocol):
        def column(self, *args: Any, **kwds: Any) -> NativeArrowSeries: ...
        @property
        def columns(self) -> Sequence[NativeArrowSeries]: ...

    P = ParamSpec("P")

    class UnaryFunctionP(Protocol[P]):
        """A function wrapping at-most 1 `Expr` input."""

        def __call__(
            self, native: ChunkedOrScalarAny, /, *args: P.args, **kwds: P.kwargs
        ) -> ChunkedOrScalarAny: ...

    class VectorFunction(Protocol[P]):
        def __call__(
            self, native: ChunkedArrayAny, /, *args: P.args, **kwds: P.kwargs
        ) -> ChunkedArrayAny: ...

    class BooleanLengthPreserving(Protocol):
        def __call__(
            self, indices: ChunkedArrayAny, aggregated: ChunkedArrayAny, /
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
    "NumericOrTemporalScalarT", bound=NumericOrTemporalScalar, default="NumericScalar"
)


class UnaryFunction(Protocol[ScalarPT_contra, ScalarRT_co]):
    @overload
    def __call__(
        self, data: ScalarPT_contra, /, *args: Any, **kwds: Any
    ) -> ScalarRT_co: ...

    @overload
    def __call__(
        self, data: ChunkedArray[ScalarPT_contra], /, *args: Any, **kwds: Any
    ) -> ChunkedArray[ScalarRT_co]: ...

    @overload
    def __call__(
        self, data: ChunkedOrScalar[ScalarPT_contra], /, *args: Any, **kwds: Any
    ) -> ChunkedOrScalar[ScalarRT_co]: ...
    @overload
    def __call__(
        self, data: Array[ScalarPT_contra], /, *args: Any, **kwds: Any
    ) -> Array[ScalarRT_co]: ...
    @overload
    def __call__(
        self, data: ChunkedOrArray[ScalarPT_contra], /, *args: Any, **kwds: Any
    ) -> ChunkedOrArray[ScalarRT_co]: ...

    def __call__(
        self, data: Arrow[ScalarPT_contra], /, *args: Any, **kwds: Any
    ) -> Arrow[ScalarRT_co]: ...


class BinaryFunction(Protocol[ScalarPT_contra, ScalarRT_co]):
    @overload
    def __call__(
        self, x: ChunkedArray[ScalarPT_contra], y: ChunkedArray[ScalarPT_contra], /
    ) -> ChunkedArray[ScalarRT_co]: ...
    @overload
    def __call__(
        self, x: Array[ScalarPT_contra], y: Array[ScalarPT_contra], /
    ) -> Array[ScalarRT_co]: ...
    @overload
    def __call__(self, x: ScalarPT_contra, y: ScalarPT_contra, /) -> ScalarRT_co: ...
    @overload
    def __call__(
        self, x: ChunkedArray[ScalarPT_contra], y: ScalarPT_contra, /
    ) -> ChunkedArray[ScalarRT_co]: ...
    @overload
    def __call__(
        self, x: Array[ScalarPT_contra], y: ScalarPT_contra, /
    ) -> Array[ScalarRT_co]: ...
    @overload
    def __call__(
        self, x: ScalarPT_contra, y: ChunkedArray[ScalarPT_contra], /
    ) -> ChunkedArray[ScalarRT_co]: ...
    @overload
    def __call__(
        self, x: ScalarPT_contra, y: Array[ScalarPT_contra], /
    ) -> Array[ScalarRT_co]: ...
    @overload
    def __call__(
        self, x: ChunkedArray[ScalarPT_contra], y: Array[ScalarPT_contra], /
    ) -> ChunkedArray[ScalarRT_co]: ...
    @overload
    def __call__(
        self, x: Array[ScalarPT_contra], y: ChunkedArray[ScalarPT_contra], /
    ) -> ChunkedArray[ScalarRT_co]: ...
    @overload
    def __call__(
        self, x: ChunkedOrScalar[ScalarPT_contra], y: ChunkedOrScalar[ScalarPT_contra], /
    ) -> ChunkedOrScalar[ScalarRT_co]: ...

    @overload
    def __call__(
        self, x: Arrow[ScalarPT_contra], y: Arrow[ScalarPT_contra], /
    ) -> Arrow[ScalarRT_co]: ...

    def __call__(
        self, x: Arrow[ScalarPT_contra], y: Arrow[ScalarPT_contra], /
    ) -> Arrow[ScalarRT_co]: ...


class BinaryComp(
    BinaryFunction[ScalarPT_contra, "pa.BooleanScalar"], Protocol[ScalarPT_contra]
): ...


class BinaryLogical(BinaryFunction["BooleanScalar", "pa.BooleanScalar"], Protocol): ...


BinaryNumericTemporal: TypeAlias = BinaryFunction[
    NumericOrTemporalScalarT, NumericOrTemporalScalarT
]
UnaryNumeric: TypeAlias = UnaryFunction["NumericScalar", "NumericScalar"]
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
ChunkedOrScalarT = TypeVar("ChunkedOrScalarT", ChunkedArrayAny, ScalarAny)
Indices: TypeAlias = "_SizedMultiIndexSelector[ChunkedOrArray[pc.IntegerScalar]]"

ChunkedStruct: TypeAlias = "ChunkedArray[pa.StructScalar]"
StructArray: TypeAlias = "pa.StructArray | Array[pa.StructScalar]"
ChunkedList: TypeAlias = "ChunkedArray[ListScalar[DataTypeT_co]]"
ListArray: TypeAlias = "Array[ListScalar[DataTypeT_co]]"

Arrow: TypeAlias = "ChunkedOrScalar[ScalarT_co] | Array[ScalarT_co]"
ArrowAny: TypeAlias = "ChunkedOrScalarAny | ArrayAny"
SameArrowT = TypeVar("SameArrowT", ChunkedArrayAny, ArrayAny, ScalarAny)
ArrowT = TypeVar("ArrowT", bound=ArrowAny)
ArrowListT = TypeVar("ArrowListT", bound="Arrow[ListScalar[Any]]")
Predicate: TypeAlias = "Arrow[BooleanScalar]"
"""Any `pyarrow` container that wraps boolean."""

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

SearchSortedSide: TypeAlias = Literal["left", "right"]
