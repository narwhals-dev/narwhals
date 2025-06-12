from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    cast,
    overload,
)

import polars as pl

from narwhals._polars.utils import (
    catch_polars_exception,
    extract_args_kwargs,
    extract_native,
    narwhals_to_native_dtype,
    native_to_narwhals_dtype,
)
from narwhals._utils import Implementation, requires, validate_backend_version
from narwhals.dependencies import is_numpy_array_1d

if TYPE_CHECKING:
    from types import ModuleType
    from typing import TypeVar

    import pandas as pd
    import pyarrow as pa
    from typing_extensions import Self, TypeIs

    from narwhals._polars.dataframe import Method, PolarsDataFrame
    from narwhals._polars.expr import PolarsExpr
    from narwhals._polars.namespace import PolarsNamespace
    from narwhals._utils import Version, _FullContext
    from narwhals.dtypes import DType
    from narwhals.series import Series
    from narwhals.typing import Into1DArray, IntoDType, MultiIndexSelector, _1DArray

    T = TypeVar("T")


# Series methods where PolarsSeries just defers to Polars.Series directly.
INHERITED_METHODS = frozenset(
    [
        "__add__",
        "__and__",
        "__floordiv__",
        "__invert__",
        "__iter__",
        "__mod__",
        "__mul__",
        "__or__",
        "__pow__",
        "__radd__",
        "__rand__",
        "__rfloordiv__",
        "__rmod__",
        "__rmul__",
        "__ror__",
        "__rsub__",
        "__rtruediv__",
        "__sub__",
        "__truediv__",
        "abs",
        "all",
        "any",
        "arg_max",
        "arg_min",
        "arg_true",
        "clip",
        "count",
        "cum_max",
        "cum_min",
        "cum_prod",
        "cum_sum",
        "diff",
        "drop_nulls",
        "exp",
        "fill_null",
        "filter",
        "gather_every",
        "head",
        "is_between",
        "is_finite",
        "is_first_distinct",
        "is_in",
        "is_last_distinct",
        "is_null",
        "is_sorted",
        "is_unique",
        "item",
        "len",
        "log",
        "max",
        "mean",
        "min",
        "mode",
        "n_unique",
        "null_count",
        "quantile",
        "rank",
        "round",
        "sample",
        "shift",
        "skew",
        "std",
        "sum",
        "tail",
        "to_arrow",
        "to_frame",
        "to_list",
        "to_pandas",
        "unique",
        "var",
        "zip_with",
    ]
)


class PolarsSeries:
    def __init__(
        self, series: pl.Series, *, backend_version: tuple[int, ...], version: Version
    ) -> None:
        self._native_series: pl.Series = series
        self._backend_version = backend_version
        self._implementation = Implementation.POLARS
        self._version = version
        validate_backend_version(self._implementation, self._backend_version)

    def __repr__(self) -> str:  # pragma: no cover
        return "PolarsSeries"

    def __narwhals_namespace__(self) -> PolarsNamespace:
        from narwhals._polars.namespace import PolarsNamespace

        return PolarsNamespace(
            backend_version=self._backend_version, version=self._version
        )

    def __narwhals_series__(self) -> Self:
        return self

    def __native_namespace__(self) -> ModuleType:
        if self._implementation is Implementation.POLARS:
            return self._implementation.to_native_namespace()

        msg = f"Expected polars, got: {type(self._implementation)}"  # pragma: no cover
        raise AssertionError(msg)

    def _with_version(self, version: Version) -> Self:
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
        dtype: IntoDType | None = None,
    ) -> Self:
        version = context._version
        backend_version = context._backend_version
        dtype_pl = (
            narwhals_to_native_dtype(dtype, version, backend_version) if dtype else None
        )
        # NOTE: `Iterable` is fine, annotation is overly narrow
        # https://github.com/pola-rs/polars/blob/82d57a4ee41f87c11ca1b1af15488459727efdd7/py-polars/polars/series/series.py#L332-L333
        native = pl.Series(name=name, values=cast("Sequence[Any]", data), dtype=dtype_pl)
        return cls.from_native(native, context=context)

    @staticmethod
    def _is_native(obj: pl.Series | Any) -> TypeIs[pl.Series]:
        return isinstance(obj, pl.Series)

    @classmethod
    def from_native(cls, data: pl.Series, /, *, context: _FullContext) -> Self:
        return cls(
            data, backend_version=context._backend_version, version=context._version
        )

    @classmethod
    def from_numpy(cls, data: Into1DArray, /, *, context: _FullContext) -> Self:
        native = pl.Series(data if is_numpy_array_1d(data) else [data])
        return cls.from_native(native, context=context)

    def to_narwhals(self) -> Series[pl.Series]:
        return self._version.series(self, level="full")

    def _with_native(self, series: pl.Series) -> Self:
        return self.__class__(
            series, backend_version=self._backend_version, version=self._version
        )

    @overload
    def _from_native_object(self, series: pl.Series) -> Self: ...

    @overload
    def _from_native_object(self, series: pl.DataFrame) -> PolarsDataFrame: ...

    @overload
    def _from_native_object(self, series: T) -> T: ...

    def _from_native_object(
        self, series: pl.Series | pl.DataFrame | T
    ) -> Self | PolarsDataFrame | T:
        if self._is_native(series):
            return self._with_native(series)
        if isinstance(series, pl.DataFrame):
            from narwhals._polars.dataframe import PolarsDataFrame

            return PolarsDataFrame.from_native(series, context=self)
        # scalar
        return series

    def _to_expr(self) -> PolarsExpr:
        return self.__narwhals_namespace__()._expr._from_series(self)

    def __getattr__(self, attr: str) -> Any:
        if attr not in INHERITED_METHODS:
            msg = f"{self.__class__.__name__} has not attribute '{attr}'."
            raise AttributeError(msg)

        def func(*args: Any, **kwargs: Any) -> Any:
            pos, kwds = extract_args_kwargs(args, kwargs)
            return self._from_native_object(getattr(self.native, attr)(*pos, **kwds))

        return func

    def __len__(self) -> int:
        return len(self.native)

    @property
    def name(self) -> str:
        return self.native.name

    @property
    def dtype(self) -> DType:
        return native_to_narwhals_dtype(
            self.native.dtype, self._version, self._backend_version
        )

    @property
    def native(self) -> pl.Series:
        return self._native_series

    def alias(self, name: str) -> Self:
        return self._from_native_object(self.native.alias(name))

    def __getitem__(self, item: MultiIndexSelector[Self]) -> Any | Self:
        if isinstance(item, PolarsSeries):
            return self._from_native_object(self.native.__getitem__(item.native))
        return self._from_native_object(self.native.__getitem__(item))

    def cast(self, dtype: IntoDType) -> Self:
        dtype_pl = narwhals_to_native_dtype(dtype, self._version, self._backend_version)
        return self._with_native(self.native.cast(dtype_pl))

    @requires.backend_version((1,))
    def replace_strict(
        self,
        old: Sequence[Any] | Mapping[Any, Any],
        new: Sequence[Any],
        *,
        return_dtype: IntoDType | None,
    ) -> Self:
        ser = self.native
        dtype = (
            narwhals_to_native_dtype(return_dtype, self._version, self._backend_version)
            if return_dtype
            else None
        )
        return self._with_native(ser.replace_strict(old, new, return_dtype=dtype))

    def to_numpy(self, dtype: Any = None, *, copy: bool | None = None) -> _1DArray:
        return self.__array__(dtype, copy=copy)

    def __array__(self, dtype: Any, *, copy: bool | None) -> _1DArray:
        if self._backend_version < (0, 20, 29):
            return self.native.__array__(dtype=dtype)
        return self.native.__array__(dtype=dtype, copy=copy)

    def __eq__(self, other: object) -> Self:  # type: ignore[override]
        return self._with_native(self.native.__eq__(extract_native(other)))

    def __ne__(self, other: object) -> Self:  # type: ignore[override]
        return self._with_native(self.native.__ne__(extract_native(other)))

    # NOTE: `pyright` is being reasonable here
    def __ge__(self, other: Any) -> Self:
        return self._with_native(self.native.__ge__(extract_native(other)))  # pyright: ignore[reportArgumentType]

    def __gt__(self, other: Any) -> Self:
        return self._with_native(self.native.__gt__(extract_native(other)))  # pyright: ignore[reportArgumentType]

    def __le__(self, other: Any) -> Self:
        return self._with_native(self.native.__le__(extract_native(other)))  # pyright: ignore[reportArgumentType]

    def __lt__(self, other: Any) -> Self:
        return self._with_native(self.native.__lt__(extract_native(other)))  # pyright: ignore[reportArgumentType]

    def __rpow__(self, other: PolarsSeries | Any) -> Self:
        result = self.native.__rpow__(extract_native(other))
        if self._backend_version < (1, 16, 1):
            # Explicitly set alias to work around https://github.com/pola-rs/polars/issues/20071
            result = result.alias(self.name)
        return self._with_native(result)

    def is_nan(self) -> Self:
        try:
            native_is_nan = self.native.is_nan()
        except Exception as e:  # noqa: BLE001
            raise catch_polars_exception(e, self._backend_version) from None
        if self._backend_version < (1, 18):  # pragma: no cover
            select = pl.when(self.native.is_not_null()).then(native_is_nan)
            return self._with_native(pl.select(select)[self.name])
        return self._with_native(native_is_nan)

    def median(self) -> Any:
        from narwhals.exceptions import InvalidOperationError

        if not self.dtype.is_numeric():
            msg = "`median` operation not supported for non-numeric input type."
            raise InvalidOperationError(msg)

        return self.native.median()

    def to_dummies(self, *, separator: str, drop_first: bool) -> PolarsDataFrame:
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
        return PolarsDataFrame.from_native(result, context=self)

    def ewm_mean(
        self,
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

    @requires.backend_version((1,))
    def rolling_var(
        self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
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

    @requires.backend_version((1,))
    def rolling_std(
        self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
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

    def rolling_sum(self, window_size: int, *, min_samples: int, center: bool) -> Self:
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

    def rolling_mean(self, window_size: int, *, min_samples: int, center: bool) -> Self:
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

    def sort(self, *, descending: bool, nulls_last: bool) -> Self:
        if self._backend_version < (0, 20, 6):
            result = self.native.sort(descending=descending)

            if nulls_last:
                is_null = result.is_null()
                result = pl.concat([result.filter(~is_null), result.filter(is_null)])
        else:
            result = self.native.sort(descending=descending, nulls_last=nulls_last)

        return self._with_native(result)

    def scatter(self, indices: int | Sequence[int], values: Any) -> Self:
        s = self.native.clone().scatter(indices, extract_native(values))
        return self._with_native(s)

    def value_counts(
        self, *, sort: bool, parallel: bool, name: str | None, normalize: bool
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
        return PolarsDataFrame.from_native(result, context=self)

    def cum_count(self, *, reverse: bool) -> Self:
        if self._backend_version < (0, 20, 4):
            not_null_series = ~self.native.is_null()
            result = not_null_series.cum_sum(reverse=reverse)
        else:
            result = self.native.cum_count(reverse=reverse)

        return self._with_native(result)

    def __contains__(self, other: Any) -> bool:
        try:
            return self.native.__contains__(other)
        except Exception as e:  # noqa: BLE001
            raise catch_polars_exception(e, self._backend_version) from None

    def hist(  # noqa: C901, PLR0912
        self,
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
            return PolarsDataFrame.from_native(pl.DataFrame(data), context=self)

        if self.native.count() < 1:
            data_dict: dict[str, Sequence[Any] | pl.Series]
            if bins is not None:
                data_dict = {
                    "breakpoint": bins[1:],
                    "count": pl.zeros(n=len(bins) - 1, dtype=pl.Int64, eager=True),
                }
            elif (bin_count is not None) and bin_count == 1:
                data_dict = {"breakpoint": [1.0], "count": [0]}
            elif (bin_count is not None) and bin_count > 1:
                data_dict = {
                    "breakpoint": pl.int_range(1, bin_count + 1, eager=True) / bin_count,
                    "count": pl.zeros(n=bin_count, dtype=pl.Int64, eager=True),
                }
            else:  # pragma: no cover
                msg = (
                    "congratulations, you entered unreachable code - please report a bug"
                )
                raise AssertionError(msg)
            if not include_breakpoint:
                del data_dict["breakpoint"]
            return PolarsDataFrame.from_native(pl.DataFrame(data_dict), context=self)

        # polars <1.15 does not adjust the bins when they have equivalent min/max
        # polars <1.5 with bin_count=...
        # returns bins that range from -inf to +inf and has bin_count + 1 bins.
        #   for compat: convert `bin_count=` call to `bins=`
        if (self._backend_version < (1, 15)) and (
            bin_count is not None
        ):  # pragma: no cover
            lower = cast("float", self.native.min())
            upper = cast("float", self.native.max())
            if lower == upper:
                width = 1 / bin_count
                lower -= 0.5
                upper += 0.5
            else:
                width = (upper - lower) / bin_count

            bins = (pl.int_range(0, bin_count + 1, eager=True) * width + lower).to_list()
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

        if self._backend_version < (1, 0) and include_breakpoint:
            df = df.rename({"break_point": "breakpoint"})

        #  polars<1.15 implicitly adds -inf and inf to either end of bins
        if self._backend_version < (1, 15) and bins is not None:  # pragma: no cover
            r = pl.int_range(0, len(df))
            df = df.filter((r > 0) & (r < len(df) - 1))

        # polars<1.27 makes the lowest bin a left/right closed interval.
        if self._backend_version < (1, 27) and bins is not None:
            df[0, "count"] += (series == bins[0]).sum()

        return PolarsDataFrame.from_native(df, context=self)

    def to_polars(self) -> pl.Series:
        return self.native

    @property
    def dt(self) -> PolarsSeriesDateTimeNamespace:
        return PolarsSeriesDateTimeNamespace(self)

    @property
    def str(self) -> PolarsSeriesStringNamespace:
        return PolarsSeriesStringNamespace(self)

    @property
    def cat(self) -> PolarsSeriesCatNamespace:
        return PolarsSeriesCatNamespace(self)

    @property
    def struct(self) -> PolarsSeriesStructNamespace:
        return PolarsSeriesStructNamespace(self)

    __add__: Method[Self]
    __and__: Method[Self]
    __floordiv__: Method[Self]
    __invert__: Method[Self]
    __iter__: Method[Iterator[Any]]
    __mod__: Method[Self]
    __mul__: Method[Self]
    __or__: Method[Self]
    __pow__: Method[Self]
    __radd__: Method[Self]
    __rand__: Method[Self]
    __rfloordiv__: Method[Self]
    __rmod__: Method[Self]
    __rmul__: Method[Self]
    __ror__: Method[Self]
    __rsub__: Method[Self]
    __rtruediv__: Method[Self]
    __sub__: Method[Self]
    __truediv__: Method[Self]
    abs: Method[Self]
    all: Method[bool]
    any: Method[bool]
    arg_max: Method[int]
    arg_min: Method[int]
    arg_true: Method[Self]
    clip: Method[Self]
    count: Method[int]
    cum_max: Method[Self]
    cum_min: Method[Self]
    cum_prod: Method[Self]
    cum_sum: Method[Self]
    diff: Method[Self]
    drop_nulls: Method[Self]
    exp: Method[Self]
    fill_null: Method[Self]
    filter: Method[Self]
    gather_every: Method[Self]
    head: Method[Self]
    is_between: Method[Self]
    is_finite: Method[Self]
    is_first_distinct: Method[Self]
    is_in: Method[Self]
    is_last_distinct: Method[Self]
    is_null: Method[Self]
    is_sorted: Method[bool]
    is_unique: Method[Self]
    item: Method[Any]
    len: Method[int]
    log: Method[Self]
    max: Method[Any]
    mean: Method[float]
    min: Method[Any]
    mode: Method[Self]
    n_unique: Method[int]
    null_count: Method[int]
    quantile: Method[float]
    rank: Method[Self]
    round: Method[Self]
    sample: Method[Self]
    shift: Method[Self]
    skew: Method[float | None]
    std: Method[float]
    sum: Method[float]
    tail: Method[Self]
    to_arrow: Method[pa.Array[Any]]
    to_frame: Method[PolarsDataFrame]
    to_list: Method[list[Any]]
    to_pandas: Method[pd.Series[Any]]
    unique: Method[Self]
    var: Method[float]
    zip_with: Method[Self]

    @property
    def list(self) -> PolarsSeriesListNamespace:
        return PolarsSeriesListNamespace(self)


class PolarsSeriesDateTimeNamespace:
    def __init__(self, series: PolarsSeries) -> None:
        self._compliant_series = series

    def __getattr__(self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            pos, kwds = extract_args_kwargs(args, kwargs)
            return self._compliant_series._with_native(
                getattr(self._compliant_series.native.dt, attr)(*pos, **kwds)
            )

        return func


class PolarsSeriesStringNamespace:
    def __init__(self, series: PolarsSeries) -> None:
        self._compliant_series = series

    def __getattr__(self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            pos, kwds = extract_args_kwargs(args, kwargs)
            return self._compliant_series._with_native(
                getattr(self._compliant_series.native.str, attr)(*pos, **kwds)
            )

        return func


class PolarsSeriesCatNamespace:
    def __init__(self, series: PolarsSeries) -> None:
        self._compliant_series = series

    def __getattr__(self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            pos, kwds = extract_args_kwargs(args, kwargs)
            return self._compliant_series._with_native(
                getattr(self._compliant_series.native.cat, attr)(*pos, **kwds)
            )

        return func


class PolarsSeriesListNamespace:
    def __init__(self, series: PolarsSeries) -> None:
        self._series = series

    def len(self) -> PolarsSeries:
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
    def __getattr__(self, attr: str) -> Any:  # pragma: no cover
        def func(*args: Any, **kwargs: Any) -> Any:
            pos, kwds = extract_args_kwargs(args, kwargs)
            return self._series._with_native(
                getattr(self._series.native.list, attr)(*pos, **kwds)
            )

        return func


class PolarsSeriesStructNamespace:
    def __init__(self, series: PolarsSeries) -> None:
        self._compliant_series = series

    def __getattr__(self, attr: str) -> Any:
        def func(*args: Any, **kwargs: Any) -> Any:
            pos, kwds = extract_args_kwargs(args, kwargs)
            return self._compliant_series._with_native(
                getattr(self._compliant_series.native.struct, attr)(*pos, **kwds)
            )

        return func
