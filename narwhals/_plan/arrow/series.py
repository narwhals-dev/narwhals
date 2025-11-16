from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow.compute as pc

from narwhals._arrow.utils import narwhals_to_native_dtype, native_to_narwhals_dtype
from narwhals._plan.arrow import functions as fn, options
from narwhals._plan.arrow.common import ArrowFrameSeries as FrameSeries
from narwhals._plan.compliant.series import CompliantSeries
from narwhals._plan.compliant.typing import namespace
from narwhals._utils import Version, generate_repr
from narwhals.dependencies import is_numpy_array_1d

if TYPE_CHECKING:
    from collections.abc import Iterable

    import polars as pl
    from typing_extensions import Self

    from narwhals._plan.arrow.dataframe import ArrowDataFrame as DataFrame
    from narwhals._plan.arrow.typing import ChunkedArrayAny
    from narwhals.dtypes import DType
    from narwhals.typing import Into1DArray, IntoDType, _1DArray


class ArrowSeries(FrameSeries["ChunkedArrayAny"], CompliantSeries["ChunkedArrayAny"]):
    _name: str

    def __repr__(self) -> str:
        return generate_repr(f"nw.{type(self).__name__}", self.native.__repr__())

    def _with_native(self, native: ChunkedArrayAny) -> Self:
        return self.from_native(native, self.name, version=self.version)

    def to_frame(self) -> DataFrame:
        return namespace(self)._dataframe.from_dict({self.name: self.native})

    def to_list(self) -> list[Any]:
        return self.native.to_pylist()

    def to_numpy(self, dtype: Any = None, *, copy: bool | None = None) -> _1DArray:
        return self.native.to_numpy()

    def to_polars(self) -> pl.Series:
        import polars as pl  # ignore-banned-import
        # NOTE: Recommended in https://github.com/pola-rs/polars/issues/22921#issuecomment-2908506022

        return pl.Series(self.native)

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

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Self:
        opts = options.array_sort(descending=descending, nulls_last=nulls_last)
        indices = pc.array_sort_indices(self.native, options=opts)
        return self._with_native(self._gather(indices))

    def scatter(self, indices: Self, values: Self) -> Self:
        mask = fn.is_in(fn.int_range(len(self), chunked=False), indices.native)
        replacements = fn.array(values._gather(pc.sort_indices(indices.native)))
        return self._with_native(pc.replace_with_mask(self.native, mask, replacements))

    def is_in(self, other: Self) -> Self:
        return self._with_native(fn.is_in(self.native, other.native))

    def has_nulls(self) -> bool:
        return bool(self.native.null_count)
