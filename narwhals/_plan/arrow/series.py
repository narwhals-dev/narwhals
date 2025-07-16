from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._arrow.utils import narwhals_to_native_dtype, native_to_narwhals_dtype
from narwhals._plan.arrow import functions as fn
from narwhals._plan.protocols import DummyCompliantSeries
from narwhals._utils import Version
from narwhals.dependencies import is_numpy_array_1d

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Self

    from narwhals._arrow.typing import ChunkedArrayAny  # noqa: F401
    from narwhals.dtypes import DType
    from narwhals.typing import Into1DArray, IntoDType


class ArrowSeries(DummyCompliantSeries["ChunkedArrayAny"]):
    def to_list(self) -> list[Any]:
        return self.native.to_pylist()

    def __len__(self) -> int:
        return self.native.length()

    @property
    def dtype(self) -> DType:
        return native_to_narwhals_dtype(self.native.type, self._version)

    @classmethod
    def from_numpy(
        cls, data: Into1DArray, name: str = "", /, *, version: Version = Version.MAIN
    ) -> Self:
        return cls.from_iterable(
            data if is_numpy_array_1d(data) else [data], name=name, version=version
        )

    @classmethod
    def from_iterable(
        cls,
        data: Iterable[Any],
        *,
        version: Version,
        name: str = "",
        dtype: IntoDType | None = None,
    ) -> Self:
        dtype_pa = narwhals_to_native_dtype(dtype, version) if dtype else None
        return cls.from_native(fn.chunked_array([data], dtype_pa), name, version=version)

    def cast(self, dtype: IntoDType) -> Self:
        dtype_pa = narwhals_to_native_dtype(dtype, self.version)
        return self._with_native(fn.cast(self.native, dtype_pa))
