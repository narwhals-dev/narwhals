from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Sequence
from typing import cast
from typing import overload

import polars as pl

from narwhals._polars.utils import catch_polars_exception
from narwhals._polars.utils import extract_args_kwargs
from narwhals._polars.utils import extract_native
from narwhals._polars.utils import narwhals_to_native_dtype
from narwhals._polars.utils import native_to_narwhals_dtype
from narwhals.dependencies import is_numpy_array_1d
from narwhals.utils import Implementation
from narwhals.utils import validate_backend_version

if TYPE_CHECKING:
    from types import ModuleType
    from typing import TypeVar

    from typing_extensions import Self

    from narwhals._polars.dataframe import PolarsDataFrame
    from narwhals._polars.expr import PolarsExpr
    from narwhals._polars.namespace import PolarsNamespace
    from narwhals.dtypes import DType
    from narwhals.typing import Into1DArray
    from narwhals.typing import _1DArray
    from narwhals.utils import Version
    from narwhals.utils import _FullContext

    T = TypeVar("T")


class PolarsSeries:
    def __init__(
        self: Self,
        series: pl.Series,
        *,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> None:
        self._native_series: pl.Series = series
        self._backend_version = backend_version
        self._implementation = Implementation.POLARS
        self._version = version
        validate_backend_version(self._implementation, self._backend_version)

    def __repr__(self: Self) -> str:  # pragma: no cover
        return "PolarsSeries"

    def __narwhals_namespace__(self) -> PolarsNamespace:
        from narwhals._polars.namespace import PolarsNamespace

        return PolarsNamespace(
            backend_version=self._backend_version, version=self._version
        )

    def __narwhals_series__(self: Self) -> Self:
        return self

    def __native_namespace__(self: Self) -> ModuleType:
        if self._implementation is Implementation.POLARS:
            return self._implementation.to_native_namespace()

        msg = f"Expected polars, got: {type(self._implementation)}"  # pragma: no cover
        raise AssertionError(msg)

    def _with_version(self: Self, version: Version) -> Self:
        return self.__class__(
            self.native, backend_version=self._backend_version, version=version
        )

    @classmethod
    def from_iterable(
        cls,
        data: Iterable[Any],
        *,
        context: _FullContext,
        name: str = "",
        dtype: DType | type[DType] | None = None,
    ) -> Self:
        version = context._version
        backend_version = context._backend_version
        dtype_pl = (
            narwhals_to_native_dtype(dtype, version, backend_version) if dtype else None
        )
        # NOTE: `Iterable` is fine, annotation is overly narrow
        # https://github.com/pola-rs/polars/blob/82d57a4ee41f87c11ca1b1af15488459727efdd7/py-polars/polars/series/series.py#L332-L333
        return cls(
            pl.Series(name=name, values=cast("Sequence[Any]", data), dtype=dtype_pl),
            backend_version=backend_version,
            version=version,
        )

    @classmethod
    def from_numpy(cls, data: Into1DArray, /, *, context: _FullContext) -> Self:
        return cls(
            pl.Series(data if is_numpy_array_1d(data) else [data]),
            backend_version=context._backend_version,
            version=context._version,
        )

    def _with_native(self: Self, series: pl.Series) -> Self:
        return self.__class__(
            series, backend_version=self._backend_version, version=self._version
        )

    @overload
    def _from_native_object(self: Self, series: pl.Series) -> Self: ...

    @overload
    def _from_native_object(self: Self, series: pl.DataFrame) -> PolarsDataFrame: ...

    @overload
    def _from_native_object(self: Self, series: T) -> T: ...

    def _from_native_object(
        self: Self, series: pl.Series | pl.DataFrame | T
    ) -> Self | PolarsDataFrame | T:
        if isinstance(series, pl.Series):
            return self._with_native(series)
        if isinstance(series, pl.DataFrame):
            from narwhals._polars.dataframe import PolarsDataFrame

            return PolarsDataFrame(
                series, backend_version=self._backend_version, version=self._version
            )
        # scalar
        return series

    def _to_expr(self) -> PolarsExpr:
        return self.__narwhals_namespace__()._expr._from_series(self)

    def __getattr__(self: Self, attr: str) -> Any:
        if attr == "as_py":  # pragma: no cover
            raise AttributeError

        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return self._from_native_object(getattr(self.native, attr)(*args, **kwargs))

        return func

    def __len__(self: Self) -> int:
        return len(self.native)

    @property
    def name(self: Self) -> str:
        return self.native.name

    @property
    def dtype(self: Self) -> DType:
        return native_to_narwhals_dtype(
            self.native.dtype, self._version, self._backend_version
        )

    @property
    def native(self) -> pl.Series:
        return self._native_series

    def alias(self, name: str) -> Self:
        return self._from_native_object(self.native.alias(name))

    @overload
    def __getitem__(self: Self, item: int) -> Any: ...

    @overload
    def __getitem__(self: Self, item: slice | Sequence[int] | pl.Series) -> Self: ...

    def __getitem__(
        self: Self, item: int | slice | Sequence[int] | pl.Series
    ) -> Any | Self:
        return self._from_native_object(self.native.__getitem__(item))

    def cast(self: Self, dtype: DType) -> Self:
        dtype_pl = narwhals_to_native_dtype(dtype, self._version, self._backend_version)
        return self._with_native(self.native.cast(dtype_pl))

    def replace_strict(
        self: Self, old: Sequence[Any], new: Sequence[Any], *, return_dtype: DType | None
    ) -> Self:
        ser = self.native
        dtype = (
            narwhals_to_native_dtype(return_dtype, self._version, self._backend_version)
            if return_dtype
            else None
        )
        if self._backend_version < (1,):
            msg = f"`replace_strict` is only available in Polars>=1.0, found version {self._backend_version}"
            raise NotImplementedError(msg)
        return self._with_native(ser.replace_strict(old, new, return_dtype=dtype))

    def to_numpy(self, dtype: Any = None, *, copy: bool | None = None) -> _1DArray:
        return self.__array__(dtype, copy=copy)

    def __array__(self: Self, dtype: Any, *, copy: bool | None) -> _1DArray:
        if self._backend_version < (0, 20, 29):
            return self.native.__array__(dtype=dtype)
        return self.native.__array__(dtype=dtype, copy=copy)

    def __eq__(self: Self, other: object) -> Self:  # type: ignore[override]
        return self._with_native(self.native.__eq__(extract_native(other)))

    def __ne__(self: Self, other: object) -> Self:  # type: ignore[override]
        return self._with_native(self.native.__ne__(extract_native(other)))

    def __ge__(self: Self, other: Any) -> Self:
        return self._with_native(self.native.__ge__(extract_native(other)))

    def __gt__(self: Self, other: Any) -> Self:
        return self._with_native(self.native.__gt__(extract_native(other)))

    def __le__(self: Self, other: Any) -> Self:
        return self._with_native(self.native.__le__(extract_native(other)))

    def __lt__(self: Self, other: Any) -> Self:
        return self._with_native(self.native.__lt__(extract_native(other)))

    def __and__(self: Self, other: PolarsSeries | bool | Any) -> Self:
        return self._with_native(self.native.__and__(extract_native(other)))

    def __or__(self: Self, other: PolarsSeries | bool | Any) -> Self:
        return self._with_native(self.native.__or__(extract_native(other)))

    def __add__(self: Self, other: PolarsSeries | Any) -> Self:
        return self._with_native(self.native.__add__(extract_native(other)))

    def __radd__(self: Self, other: PolarsSeries | Any) -> Self:
        return self._with_native(self.native.__radd__(extract_native(other)))

    def __sub__(self: Self, other: PolarsSeries | Any) -> Self:
        return self._with_native(self.native.__sub__(extract_native(other)))

    def __rsub__(self: Self, other: PolarsSeries | Any) -> Self:
        return self._with_native(self.native.__rsub__(extract_native(other)))

    def __mul__(self: Self, other: PolarsSeries | Any) -> Self:
        return self._with_native(self.native.__mul__(extract_native(other)))

    def __rmul__(self: Self, other: PolarsSeries | Any) -> Self:
        return self._with_native(self.native.__rmul__(extract_native(other)))

    def __pow__(self: Self, other: PolarsSeries | Any) -> Self:
        return self._with_native(self.native.__pow__(extract_native(other)))

    def __rpow__(self: Self, other: PolarsSeries | Any) -> Self:
        result = self.native.__rpow__(extract_native(other))
        if self._backend_version < (1, 16, 1):
            # Explicitly set alias to work around https://github.com/pola-rs/polars/issues/20071
            result = result.alias(self.name)
        return self._with_native(result)

    def __invert__(self: Self) -> Self:
        return self._with_native(self.native.__invert__())

    def is_nan(self: Self) -> Self:
        try:
            native_is_nan = self.native.is_nan()
        except Exception as e:  # noqa: BLE001
            raise catch_polars_exception(e, self._backend_version) from None
        if self._backend_version < (1, 18):  # pragma: no cover
            select = pl.when(self.native.is_not_null()).then(native_is_nan)
            return self._with_native(pl.select(select)[self.name])
        return self._with_native(native_is_nan)

    def median(self: Self) -> Any:
        from narwhals.exceptions import InvalidOperationError

        if not self.dtype.is_numeric():
            msg = "`median` operation not supported for non-numeric input type."
            raise InvalidOperationError(msg)

        return self.native.median()

    def to_dummies(self: Self, *, separator: str, drop_first: bool) -> PolarsDataFrame:
        from narwhals._polars.dataframe import PolarsDataFrame

        if self._backend_version < (0, 20, 15):
            has_nulls = self.native.is_null().any()
            result = self.native.to_dummies(separator=separator)
            output_columns = result.columns
            if drop_first:
                _ = output_columns.pop(int(has_nulls))

            result = result.select(output_columns)
        else:
            result = self.native.to_dummies(separator=separator, drop_first=drop_first)
        result = result.with_columns(pl.all().cast(pl.Int8))
        return PolarsDataFrame(
            result, backend_version=self._backend_version, version=self._version
        )

    def ewm_mean(
        self: Self,
        *,
        com: float | None,
        span: float | None,
        half_life: float | None,
        alpha: float | None,
        adjust: bool,
        min_samples: int,
        ignore_nulls: bool,
    ) -> Self:
        extra_kwargs = (
            {"min_periods": min_samples}
            if self._backend_version < (1, 21, 0)
            else {"min_samples": min_samples}
        )

        native_result = self.native.ewm_mean(
            com=com,
            span=span,
            half_life=half_life,
            alpha=alpha,
            adjust=adjust,
            ignore_nulls=ignore_nulls,
            **extra_kwargs,
        )
        if self._backend_version < (1,):  # pragma: no cover
            return self._with_native(
                pl.select(
                    pl.when(~self.native.is_null()).then(native_result).otherwise(None)
                )[self.native.name]
            )

        return self._with_native(native_result)

    def rolling_var(
        self: Self,
        window_size: int,
        *,
        min_samples: int,
        center: bool,
        ddof: int,
    ) -> Self:
        if self._backend_version < (1,):  # pragma: no cover
            msg = "`rolling_var` not implemented for polars older than 1.0"
            raise NotImplementedError(msg)

        extra_kwargs: dict[str, Any] = (
            {"min_periods": min_samples}
            if self._backend_version < (1, 21, 0)
            else {"min_samples": min_samples}
        )

        return self._with_native(
            self.native.rolling_var(
                window_size=window_size, center=center, ddof=ddof, **extra_kwargs
            )
        )

    def rolling_std(
        self: Self,
        window_size: int,
        *,
        min_samples: int,
        center: bool,
        ddof: int,
    ) -> Self:
        if self._backend_version < (1,):  # pragma: no cover
            msg = "`rolling_std` not implemented for polars older than 1.0"
            raise NotImplementedError(msg)

        extra_kwargs: dict[str, Any] = (
            {"min_periods": min_samples}
            if self._backend_version < (1, 21, 0)
            else {"min_samples": min_samples}
        )

        return self._with_native(
            self.native.rolling_std(
                window_size=window_size, center=center, ddof=ddof, **extra_kwargs
            )
        )

    def rolling_sum(
        self: Self,
        window_size: int,
        *,
        min_samples: int,
        center: bool,
    ) -> Self:
        extra_kwargs: dict[str, Any] = (
            {"min_periods": min_samples}
            if self._backend_version < (1, 21, 0)
            else {"min_samples": min_samples}
        )

        return self._with_native(
            self.native.rolling_sum(
                window_size=window_size, center=center, **extra_kwargs
            )
        )

    def rolling_mean(
        self: Self,
        window_size: int,
        *,
        min_samples: int,
        center: bool,
    ) -> Self:
        extra_kwargs: dict[str, Any] = (
            {"min_periods": min_samples}
            if self._backend_version < (1, 21, 0)
            else {"min_samples": min_samples}
        )

        return self._with_native(
            self.native.rolling_mean(
                window_size=window_size, center=center, **extra_kwargs
            )
        )

    def sort(self: Self, *, descending: bool, nulls_last: bool) -> Self:
        if self._backend_version < (0, 20, 6):
            result = self.native.sort(descending=descending)

            if nulls_last:
                is_null = result.is_null()
                result = pl.concat([result.filter(~is_null), result.filter(is_null)])
        else:
            result = self.native.sort(descending=descending, nulls_last=nulls_last)

        return self._with_native(result)

    def scatter(self: Self, indices: int | Sequence[int], values: Any) -> Self:
        s = self.native.clone().scatter(indices, extract_native(values))
        return self._with_native(s)

    def value_counts(
        self: Self,
        *,
        sort: bool,
        parallel: bool,
        name: str | None,
        normalize: bool,
    ) -> PolarsDataFrame:
        from narwhals._polars.dataframe import PolarsDataFrame

        if self._backend_version < (1, 0, 0):
            value_name_ = name or ("proportion" if normalize else "count")

            result = self.native.value_counts(sort=sort, parallel=parallel).select(
                **{
                    (self.native.name): pl.col(self.native.name),
                    value_name_: pl.col("count") / pl.sum("count")
                    if normalize
                    else pl.col("count"),
                }
            )

        else:
            result = self.native.value_counts(
                sort=sort, parallel=parallel, name=name, normalize=normalize
            )

        return PolarsDataFrame(
            result, backend_version=self._backend_version, version=self._version
        )

    def cum_count(self: Self, *, reverse: bool) -> Self:
        if self._backend_version < (0, 20, 4):
            not_null_series = ~self.native.is_null()
            result = not_null_series.cum_sum(reverse=reverse)
        else:
            result = self.native.cum_count(reverse=reverse)

        return self._with_native(result)

    def __contains__(self: Self, other: Any) -> bool:
        try:
            return self.native.__contains__(other)
        except Exception as e:  # noqa: BLE001
            raise catch_polars_exception(e, self._backend_version) from None

    def hist(
        self: Self,
        bins: list[float | int] | None,
        *,
        bin_count: int | None,
        include_breakpoint: bool,
    ) -> PolarsDataFrame:
        from narwhals._polars.dataframe import PolarsDataFrame

        if (bins is not None and len(bins) <= 1) or (bin_count == 0):  # pragma: no cover
            data: list[pl.Series] = []
            if include_breakpoint:
                data.append(pl.Series("breakpoint", [], dtype=pl.Float64))
            data.append(pl.Series("count", [], dtype=pl.UInt32))
            return PolarsDataFrame(
                pl.DataFrame(data),
                backend_version=self._backend_version,
                version=self._version,
            )
        elif (self._backend_version < (1, 15)) and self.native.count() < 1:
            data_dict: dict[str, Sequence[Any] | pl.Series]
            if bins is not None:
                data_dict = {
                    "breakpoint": bins[1:],
                    "count": pl.zeros(n=len(bins) - 1, dtype=pl.Int64, eager=True),
                }
            elif bin_count is not None:
                data_dict = {
                    "breakpoint": pl.int_range(0, bin_count, eager=True) / bin_count,
                    "count": pl.zeros(n=bin_count, dtype=pl.Int64, eager=True),
                }

            if not include_breakpoint:
                del data_dict["breakpoint"]

            return PolarsDataFrame(
                pl.DataFrame(data_dict),
                backend_version=self._backend_version,
                version=self._version,
            )

        # polars <1.15 does not adjust the bins when they have equivalent min/max
        # polars <1.5 with bin_count=...
        # returns bins that range from -inf to +inf and has bin_count + 1 bins.
        #   for compat: convert `bin_count=` call to `bins=`
        if (
            (self._backend_version < (1, 15))
            and (bin_count is not None)
            and (self.native.count() > 0)
        ):  # pragma: no cover
            lower = cast("float", self.native.min())
            upper = cast("float", self.native.max())
            pad_lowest_bin = False
            if lower == upper:
                width = 1 / bin_count
                lower -= 0.5
                upper += 0.5
            else:
                pad_lowest_bin = True
                width = (upper - lower) / bin_count

            bins = (pl.int_range(0, bin_count + 1, eager=True) * width + lower).to_list()
            if pad_lowest_bin:
                bins[0] -= 0.001 * abs(bins[0]) if bins[0] != 0 else 0.001
            bin_count = None

        # Polars inconsistently handles NaN values when computing histograms
        #   against predefined bins: https://github.com/pola-rs/polars/issues/21082
        series = self.native
        if self._backend_version < (1, 15) or bins is not None:
            series = series.set(series.is_nan(), None)

        df = series.hist(
            bins,
            bin_count=bin_count,
            include_category=False,
            include_breakpoint=include_breakpoint,
        )
        if not include_breakpoint:
            df.columns = ["count"]

        #  polars<1.15 implicitly adds -inf and inf to either end of bins
        if self._backend_version < (1, 15) and bins is not None:  # pragma: no cover
            r = pl.int_range(0, len(df))
            df = df.filter((r > 0) & (r < len(df) - 1))

        if self._backend_version < (1, 0) and include_breakpoint:
            df = df.rename({"break_point": "breakpoint"})

        return PolarsDataFrame(
            df, backend_version=self._backend_version, version=self._version
        )

    def to_polars(self: Self) -> pl.Series:
        return self.native

    @property
    def dt(self: Self) -> PolarsSeriesDateTimeNamespace:
        return PolarsSeriesDateTimeNamespace(self)

    @property
    def str(self: Self) -> PolarsSeriesStringNamespace:
        return PolarsSeriesStringNamespace(self)

    @property
    def cat(self: Self) -> PolarsSeriesCatNamespace:
        return PolarsSeriesCatNamespace(self)

    @property
    def list(self: Self) -> PolarsSeriesListNamespace:
        return PolarsSeriesListNamespace(self)

    @property
    def struct(self: Self) -> PolarsSeriesStructNamespace:
        return PolarsSeriesStructNamespace(self)


class PolarsSeriesDateTimeNamespace:
    def __init__(self: Self, series: PolarsSeries) -> None:
        self._compliant_series = series

    def __getattr__(self: Self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return self._compliant_series._with_native(
                getattr(self._compliant_series.native.dt, attr)(*args, **kwargs)
            )

        return func


class PolarsSeriesStringNamespace:
    def __init__(self: Self, series: PolarsSeries) -> None:
        self._compliant_series = series

    def __getattr__(self: Self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return self._compliant_series._with_native(
                getattr(self._compliant_series.native.str, attr)(*args, **kwargs)
            )

        return func


class PolarsSeriesCatNamespace:
    def __init__(self: Self, series: PolarsSeries) -> None:
        self._compliant_series = series

    def __getattr__(self: Self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return self._compliant_series._with_native(
                getattr(self._compliant_series.native.cat, attr)(*args, **kwargs)
            )

        return func


class PolarsSeriesListNamespace:
    def __init__(self: Self, series: PolarsSeries) -> None:
        self._series = series

    def len(self: Self) -> PolarsSeries:
        native_series = self._series.native
        native_result = native_series.list.len()

        if self._series._backend_version < (1, 16):  # pragma: no cover
            native_result = pl.select(
                pl.when(~native_series.is_null()).then(native_result).otherwise(None)
            )[native_series.name].cast(pl.UInt32())

        elif self._series._backend_version < (1, 17):  # pragma: no cover
            native_result = native_series.cast(pl.UInt32())

        return self._series._with_native(native_result)

    # TODO(FBruzzesi): Remove `pragma: no cover` once other namespace methods are added
    def __getattr__(self: Self, attr: str) -> Any:  # pragma: no cover
        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return self._series._with_native(
                getattr(self._series.native.list, attr)(*args, **kwargs)
            )

        return func


class PolarsSeriesStructNamespace:
    def __init__(self: Self, series: PolarsSeries) -> None:
        self._compliant_series = series

    def __getattr__(self: Self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = extract_args_kwargs(args, kwargs)  # type: ignore[assignment]
            return self._compliant_series._with_native(
                getattr(self._compliant_series.native.struct, attr)(*args, **kwargs)
            )

        return func
