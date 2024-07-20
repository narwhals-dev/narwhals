from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from narwhals._polars.utils import extract_args_kwargs
from narwhals._polars.utils import extract_native
from narwhals.dependencies import get_polars

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.dtypes import DType

from narwhals._polars.namespace import PolarsNamespace
from narwhals._polars.utils import reverse_translate_dtype
from narwhals._polars.utils import translate_dtype

PL = get_polars()


class PolarsSeries:
    def __init__(self, series: Any) -> None:
        self._native_series = series

    def __repr__(self) -> str:
        return "PolarsSeries"

    def __narwhals_series__(self) -> Self:
        return self

    def __native_namespace__(self) -> Any:
        return get_polars()

    def __narwhals_namespace__(self) -> PolarsNamespace:
        return PolarsNamespace()

    def _from_native_series(self, series: Any) -> Self:
        return self.__class__(series)

    def _from_native_object(self, series: Any) -> Any:
        pl = get_polars()
        if isinstance(series, pl.Series):
            return self._from_native_series(series)
        if isinstance(series, pl.DataFrame):
            from narwhals._polars.dataframe import PolarsDataFrame

            return PolarsDataFrame(series)
        if isinstance(series, pl.LazyFrame):
            from narwhals._polars.dataframe import PolarsLazyFrame

            return PolarsLazyFrame(series)
        # scalar
        return series

    def __getattr__(self, attr: str) -> Any:
        if attr == "as_py":
            raise AttributeError

        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return self._from_native_object(
                getattr(self._native_series, attr)(*args, **kwargs)
            )

        return func

    def __len__(self) -> int:
        return len(self._native_series)

    @property
    def shape(self) -> tuple[int]:
        return (len(self),)

    @property
    def name(self) -> str:
        return self._native_series.name  # type: ignore[no-any-return]

    @property
    def dtype(self) -> DType:
        return translate_dtype(self._native_series.dtype)

    def __getitem__(self, item: Any) -> Any:
        return self._from_native_object(self._native_series.__getitem__(item))

    def cast(self, dtype: DType) -> Self:
        ser = self._native_series
        dtype = reverse_translate_dtype(dtype)
        return self._from_native_series(ser.cast(dtype))

    def __eq__(self, other: object) -> Self:  # type: ignore[override]
        return self._from_native_series(self._native_series.__eq__(extract_native(other)))

    def __ne__(self, other: object) -> Self:  # type: ignore[override]
        return self._from_native_series(self._native_series.__ne__(extract_native(other)))

    def __ge__(self, other: Any) -> Self:
        return self._from_native_series(self._native_series.__ge__(extract_native(other)))

    def __gt__(self, other: Any) -> Self:
        return self._from_native_series(self._native_series.__gt__(extract_native(other)))

    def __le__(self, other: Any) -> Self:
        return self._from_native_series(self._native_series.__le__(extract_native(other)))

    def __lt__(self, other: Any) -> Self:
        return self._from_native_series(self._native_series.__lt__(extract_native(other)))

    def __and__(self, other: PolarsSeries | bool | Any) -> Self:
        return self._from_native_series(
            self._native_series.__and__(extract_native(other))
        )

    def __or__(self, other: PolarsSeries | bool | Any) -> Self:
        return self._from_native_series(self._native_series.__or__(extract_native(other)))

    def __add__(self, other: PolarsSeries | Any) -> Self:
        return self._from_native_series(
            self._native_series.__add__(extract_native(other))
        )

    def __radd__(self, other: PolarsSeries | Any) -> Self:
        return self._from_native_series(
            self._native_series.__radd__(extract_native(other))
        )

    def __sub__(self, other: PolarsSeries | Any) -> Self:
        return self._from_native_series(
            self._native_series.__sub__(extract_native(other))
        )

    def __rsub__(self, other: PolarsSeries | Any) -> Self:
        return self._from_native_series(
            self._native_series.__rsub__(extract_native(other))
        )

    def __mul__(self, other: PolarsSeries | Any) -> Self:
        return self._from_native_series(
            self._native_series.__mul__(extract_native(other))
        )

    def __rmul__(self, other: PolarsSeries | Any) -> Self:
        return self._from_native_series(
            self._native_series.__rmul__(extract_native(other))
        )

    def __pow__(self, other: PolarsSeries | Any) -> Self:
        return self._from_native_series(
            self._native_series.__pow__(extract_native(other))
        )

    def __rpow__(self, other: PolarsSeries | Any) -> Self:
        return self._from_native_series(
            self._native_series.__rpow__(extract_native(other))
        )

    def __invert__(self) -> Self:
        return self._from_native_series(self._native_series.__invert__())
