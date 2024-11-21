from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence
from typing import overload

from narwhals._polars.utils import extract_args_kwargs
from narwhals._polars.utils import extract_native
from narwhals.utils import Implementation

if TYPE_CHECKING:
    from types import ModuleType

    import numpy as np
    import polars as pl
    from typing_extensions import Self

    from narwhals._polars.dataframe import PolarsDataFrame
    from narwhals.dtypes import DType
    from narwhals.typing import DTypes

from narwhals._polars.utils import narwhals_to_native_dtype
from narwhals._polars.utils import native_to_narwhals_dtype


class PolarsSeries:
    def __init__(
        self, series: Any, *, backend_version: tuple[int, ...], dtypes: DTypes
    ) -> None:
        self._native_series: pl.Series = series
        self._backend_version = backend_version
        self._implementation = Implementation.POLARS
        self._dtypes = dtypes

    def __repr__(self) -> str:  # pragma: no cover
        return "PolarsSeries"

    def __narwhals_series__(self) -> Self:
        return self

    def __native_namespace__(self: Self) -> ModuleType:
        if self._implementation is Implementation.POLARS:
            return self._implementation.to_native_namespace()

        msg = f"Expected polars, got: {type(self._implementation)}"  # pragma: no cover
        raise AssertionError(msg)

    def _from_native_series(self, series: Any) -> Self:
        return self.__class__(
            series, backend_version=self._backend_version, dtypes=self._dtypes
        )

    def _from_native_object(self, series: Any) -> Any:
        import polars as pl  # ignore-banned-import()

        if isinstance(series, pl.Series):
            return self._from_native_series(series)
        if isinstance(series, pl.DataFrame):
            from narwhals._polars.dataframe import PolarsDataFrame

            return PolarsDataFrame(
                series, backend_version=self._backend_version, dtypes=self._dtypes
            )
        # scalar
        return series

    def __getattr__(self, attr: str) -> Any:
        if attr == "as_py":  # pragma: no cover
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
        return self._native_series.name

    @property
    def dtype(self: Self) -> DType:
        return native_to_narwhals_dtype(self._native_series.dtype, self._dtypes)

    @overload
    def __getitem__(self, item: int) -> Any: ...

    @overload
    def __getitem__(self, item: slice | Sequence[int]) -> Self: ...

    def __getitem__(self, item: int | slice | Sequence[int]) -> Any | Self:
        return self._from_native_object(self._native_series.__getitem__(item))

    def cast(self, dtype: DType) -> Self:
        ser = self._native_series
        dtype_pl = narwhals_to_native_dtype(dtype, self._dtypes)
        return self._from_native_series(ser.cast(dtype_pl))

    def replace_strict(
        self, old: Sequence[Any], new: Sequence[Any], *, return_dtype: DType | None
    ) -> Self:
        ser = self._native_series
        dtype = (
            narwhals_to_native_dtype(return_dtype, self._dtypes) if return_dtype else None
        )
        if self._backend_version < (1,):
            msg = f"`replace_strict` is only available in Polars>=1.0, found version {self._backend_version}"
            raise NotImplementedError(msg)
        return self._from_native_series(ser.replace_strict(old, new, return_dtype=dtype))

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        if self._backend_version < (0, 20, 29):
            return self._native_series.__array__(dtype=dtype)
        return self._native_series.__array__(dtype=dtype, copy=copy)

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

    def median(self) -> Any:
        from narwhals.exceptions import InvalidOperationError

        if not self.dtype.is_numeric():
            msg = "`median` operation not supported for non-numeric input type."
            raise InvalidOperationError(msg)

        return self._native_series.median()

    def to_dummies(
        self: Self, *, separator: str = "_", drop_first: bool = False
    ) -> PolarsDataFrame:
        import polars as pl  # ignore-banned-import

        from narwhals._polars.dataframe import PolarsDataFrame

        if self._backend_version < (0, 20, 15):
            has_nulls = self._native_series.is_null().any()
            result = self._native_series.to_dummies(separator=separator)
            output_columns = result.columns
            if drop_first:
                _ = output_columns.pop(int(has_nulls))

            result = result.select(output_columns)
        else:
            result = self._native_series.to_dummies(
                separator=separator, drop_first=drop_first
            )
        result = result.with_columns(pl.all().cast(pl.Int8))
        return PolarsDataFrame(
            result, backend_version=self._backend_version, dtypes=self._dtypes
        )

    def ewm_mean(
        self: Self,
        *,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
        adjust: bool = True,
        min_periods: int = 1,
        ignore_nulls: bool = False,
    ) -> Self:
        if self._backend_version < (1,):  # pragma: no cover
            msg = "`ewm_mean` not implemented for polars older than 1.0"
            raise NotImplementedError(msg)
        expr = self._native_series
        return self._from_native_series(
            expr.ewm_mean(
                com=com,
                span=span,
                half_life=half_life,
                alpha=alpha,
                adjust=adjust,
                min_periods=min_periods,
                ignore_nulls=ignore_nulls,
            )
        )

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Self:
        if self._backend_version < (0, 20, 6):
            result = self._native_series.sort(descending=descending)

            if nulls_last:
                import polars as pl  # ignore-banned-import()

                is_null = result.is_null()
                result = pl.concat([result.filter(~is_null), result.filter(is_null)])
        else:
            result = self._native_series.sort(
                descending=descending, nulls_last=nulls_last
            )

        return self._from_native_series(result)

    def scatter(self, indices: int | Sequence[int], values: Any) -> Self:
        values = extract_native(values)
        s = self._native_series.clone()
        s.scatter(indices, values)
        return self._from_native_series(s)

    def value_counts(
        self: Self,
        *,
        sort: bool = False,
        parallel: bool = False,
        name: str | None = None,
        normalize: bool = False,
    ) -> PolarsDataFrame:
        from narwhals._polars.dataframe import PolarsDataFrame

        if self._backend_version < (1, 0, 0):
            import polars as pl  # ignore-banned-import()

            value_name_ = name or ("proportion" if normalize else "count")

            result = self._native_series.value_counts(sort=sort, parallel=parallel)
            result = result.select(
                **{
                    (self._native_series.name): pl.col(self._native_series.name),
                    value_name_: pl.col("count") / pl.sum("count")
                    if normalize
                    else pl.col("count"),
                }
            )

        else:
            result = self._native_series.value_counts(
                sort=sort, parallel=parallel, name=name, normalize=normalize
            )

        return PolarsDataFrame(
            result, backend_version=self._backend_version, dtypes=self._dtypes
        )

    def cum_count(self: Self, *, reverse: bool) -> Self:
        if self._backend_version < (0, 20, 4):
            not_null_series = ~self._native_series.is_null()
            result = not_null_series.cum_sum(reverse=reverse)
        else:
            result = self._native_series.cum_count(reverse=reverse)

        return self._from_native_series(result)

    @property
    def dt(self) -> PolarsSeriesDateTimeNamespace:
        return PolarsSeriesDateTimeNamespace(self)

    @property
    def str(self) -> PolarsSeriesStringNamespace:
        return PolarsSeriesStringNamespace(self)

    @property
    def cat(self) -> PolarsSeriesCatNamespace:
        return PolarsSeriesCatNamespace(self)


class PolarsSeriesDateTimeNamespace:
    def __init__(self, series: PolarsSeries) -> None:
        self._series = series

    def __getattr__(self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return self._series._from_native_series(
                getattr(self._series._native_series.dt, attr)(*args, **kwargs)
            )

        return func


class PolarsSeriesStringNamespace:
    def __init__(self, series: PolarsSeries) -> None:
        self._series = series

    def __getattr__(self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return self._series._from_native_series(
                getattr(self._series._native_series.str, attr)(*args, **kwargs)
            )

        return func


class PolarsSeriesCatNamespace:
    def __init__(self, series: PolarsSeries) -> None:
        self._series = series

    def __getattr__(self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return self._series._from_native_series(
                getattr(self._series._native_series.cat, attr)(*args, **kwargs)
            )

        return func
