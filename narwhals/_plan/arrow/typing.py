from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, overload

from narwhals._typing_compat import TypeVar
from narwhals._utils import _StoresNative as StoresNative

if TYPE_CHECKING:
    import pyarrow as pa
    import pyarrow.compute as pc
    from typing_extensions import TypeAlias


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

    def __call__(
        self, data: ChunkedOrScalar[ScalarPT_contra], *args: Any, **kwds: Any
    ) -> ChunkedOrScalar[ScalarRT_co]: ...


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
ChunkedArray: TypeAlias = "pa.ChunkedArray[ScalarT_co]"
ChunkedOrScalar: TypeAlias = "ChunkedArray[ScalarT_co] | ScalarT_co"
ScalarAny: TypeAlias = "Scalar[Any]"
ChunkedOrScalarAny: TypeAlias = "ChunkedOrScalar[ScalarAny]"
NativeScalar: TypeAlias = ScalarAny
BinOp: TypeAlias = Callable[..., ChunkedOrScalarAny]
StoresNativeT_co = TypeVar("StoresNativeT_co", bound=StoresNative[Any], covariant=True)
