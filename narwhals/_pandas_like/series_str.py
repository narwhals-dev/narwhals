from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._compliant.any_namespace import StringNamespace
from narwhals._pandas_like.utils import PandasLikeSeriesNamespace
from narwhals._pandas_like.utils import is_pyarrow_dtype_backend
from narwhals._pandas_like.utils import to_datetime

if TYPE_CHECKING:
    from narwhals._pandas_like.series import PandasLikeSeries


class PandasLikeSeriesStringNamespace(
    PandasLikeSeriesNamespace, StringNamespace["PandasLikeSeries"]
):
    def len_chars(self) -> PandasLikeSeries:
        return self.from_native(self.native.str.len())

    def replace(
        self, pattern: str, value: str, *, literal: bool, n: int
    ) -> PandasLikeSeries:
        return self.from_native(
            self.native.str.replace(pat=pattern, repl=value, n=n, regex=not literal)
        )

    def replace_all(self, pattern: str, value: str, *, literal: bool) -> PandasLikeSeries:
        return self.replace(pattern, value, literal=literal, n=-1)

    def strip_chars(self, characters: str | None) -> PandasLikeSeries:
        return self.from_native(self.native.str.strip(characters))

    def starts_with(self, prefix: str) -> PandasLikeSeries:
        return self.from_native(self.native.str.startswith(prefix))

    def ends_with(self, suffix: str) -> PandasLikeSeries:
        return self.from_native(self.native.str.endswith(suffix))

    def contains(self, pattern: str, *, literal: bool) -> PandasLikeSeries:
        return self.from_native(self.native.str.contains(pat=pattern, regex=not literal))

    def slice(self, offset: int, length: int | None) -> PandasLikeSeries:
        stop = offset + length if length else None
        return self.from_native(self.native.str.slice(start=offset, stop=stop))

    def split(self, by: str) -> PandasLikeSeries:
        implementation = self.implementation
        if not implementation.is_cudf() and not is_pyarrow_dtype_backend(
            self.native.dtype, implementation
        ):
            msg = (
                "This operation requires a pyarrow-backed series. "
                "Please refer to https://narwhals-dev.github.io/narwhals/api-reference/narwhals/#narwhals.maybe_convert_dtypes "
                "and ensure you are using dtype_backend='pyarrow'. "
                "Additionally, make sure you have pandas version 1.5+ and pyarrow installed. "
            )
            raise TypeError(msg)
        return self.from_native(self.native.str.split(pat=by))

    def to_datetime(self, format: str | None) -> PandasLikeSeries:
        implementation = self.implementation
        if format and any(x in format for x in ("%z", "Z")):
            # We know that the inputs are timezone-aware, so we can directly pass
            # `utc=True` for better performance.
            return self.from_native(
                to_datetime(implementation, utc=True)(self.native, format=format)
            )
        result = self.from_native(
            to_datetime(implementation, utc=False)(self.native, format=format)
        )
        if (tz := getattr(result.dtype, "time_zone", None)) and tz != "UTC":
            return result.dt.convert_time_zone("UTC")
        return result

    def to_uppercase(self) -> PandasLikeSeries:
        return self.from_native(self.native.str.upper())

    def to_lowercase(self) -> PandasLikeSeries:
        return self.from_native(self.native.str.lower())
