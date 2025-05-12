from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import Sequence
from typing import cast
from typing import overload
from warnings import warn

import narwhals as nw
from narwhals import dependencies
from narwhals import exceptions
from narwhals import selectors
from narwhals.dataframe import DataFrame as NwDataFrame
from narwhals.dataframe import LazyFrame as NwLazyFrame
from narwhals.dependencies import get_polars
from narwhals.exceptions import InvalidIntoExprError
from narwhals.expr import Expr as NwExpr
from narwhals.functions import Then as NwThen
from narwhals.functions import When as NwWhen
from narwhals.functions import _from_arrow_impl
from narwhals.functions import _from_dict_impl
from narwhals.functions import _from_numpy_impl
from narwhals.functions import _new_series_impl
from narwhals.functions import _read_csv_impl
from narwhals.functions import _read_parquet_impl
from narwhals.functions import _scan_csv_impl
from narwhals.functions import _scan_parquet_impl
from narwhals.functions import get_level
from narwhals.functions import show_versions
from narwhals.functions import when as nw_when
from narwhals.schema import Schema as NwSchema
from narwhals.series import Series as NwSeries
from narwhals.stable.v1 import dtypes
from narwhals.stable.v1.dtypes import Array
from narwhals.stable.v1.dtypes import Binary
from narwhals.stable.v1.dtypes import Boolean
from narwhals.stable.v1.dtypes import Categorical
from narwhals.stable.v1.dtypes import Date
from narwhals.stable.v1.dtypes import Datetime
from narwhals.stable.v1.dtypes import Decimal
from narwhals.stable.v1.dtypes import Duration
from narwhals.stable.v1.dtypes import Enum
from narwhals.stable.v1.dtypes import Field
from narwhals.stable.v1.dtypes import Float32
from narwhals.stable.v1.dtypes import Float64
from narwhals.stable.v1.dtypes import Int8
from narwhals.stable.v1.dtypes import Int16
from narwhals.stable.v1.dtypes import Int32
from narwhals.stable.v1.dtypes import Int64
from narwhals.stable.v1.dtypes import Int128
from narwhals.stable.v1.dtypes import List
from narwhals.stable.v1.dtypes import Object
from narwhals.stable.v1.dtypes import String
from narwhals.stable.v1.dtypes import Struct
from narwhals.stable.v1.dtypes import Time
from narwhals.stable.v1.dtypes import UInt8
from narwhals.stable.v1.dtypes import UInt16
from narwhals.stable.v1.dtypes import UInt32
from narwhals.stable.v1.dtypes import UInt64
from narwhals.stable.v1.dtypes import UInt128
from narwhals.stable.v1.dtypes import Unknown
from narwhals.translate import _from_native_impl
from narwhals.translate import get_native_namespace
from narwhals.translate import to_py_scalar
from narwhals.typing import IntoDataFrameT
from narwhals.typing import IntoFrameT
from narwhals.utils import Implementation
from narwhals.utils import Version
from narwhals.utils import deprecate_native_namespace
from narwhals.utils import find_stacklevel
from narwhals.utils import generate_temporary_column_name
from narwhals.utils import inherit_doc
from narwhals.utils import is_ordered_categorical
from narwhals.utils import maybe_align_index
from narwhals.utils import maybe_convert_dtypes
from narwhals.utils import maybe_get_index
from narwhals.utils import maybe_reset_index
from narwhals.utils import maybe_set_index
from narwhals.utils import validate_strict_and_pass_though

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Mapping

    from typing_extensions import ParamSpec
    from typing_extensions import Self
    from typing_extensions import TypeVar

    from narwhals._translate import IntoArrowTable
    from narwhals.dataframe import MultiColSelector
    from narwhals.dataframe import MultiIndexSelector
    from narwhals.dtypes import DType
    from narwhals.typing import ConcatMethod
    from narwhals.typing import IntoExpr
    from narwhals.typing import IntoFrame
    from narwhals.typing import IntoLazyFrameT
    from narwhals.typing import IntoSeries
    from narwhals.typing import NonNestedLiteral
    from narwhals.typing import SingleColSelector
    from narwhals.typing import SingleIndexSelector
    from narwhals.typing import _1DArray
    from narwhals.typing import _2DArray

    FrameT = TypeVar("FrameT", "DataFrame[Any]", "LazyFrame[Any]")
    DataFrameT = TypeVar("DataFrameT", bound="DataFrame[Any]")
    LazyFrameT = TypeVar("LazyFrameT", bound="LazyFrame[Any]")
    SeriesT = TypeVar("SeriesT", bound="Series[Any]")
    IntoSeriesT = TypeVar("IntoSeriesT", bound="IntoSeries", default=Any)
    T = TypeVar("T", default=Any)
    P = ParamSpec("P")
    R = TypeVar("R")
else:
    from typing import TypeVar

    IntoSeriesT = TypeVar("IntoSeriesT", bound="IntoSeries")
    T = TypeVar("T")


class DataFrame(NwDataFrame[IntoDataFrameT]):
    @inherit_doc(NwDataFrame)
    def __init__(self, df: Any, *, level: Literal["full", "lazy", "interchange"]) -> None:
        super().__init__(df, level=level)

    # We need to override any method which don't return Self so that type
    # annotations are correct.

    @property
    def _series(self) -> type[Series[Any]]:
        return cast("type[Series[Any]]", Series)

    @property
    def _lazyframe(self) -> type[LazyFrame[Any]]:
        return cast("type[LazyFrame[Any]]", LazyFrame)

    @overload
    def __getitem__(self, item: tuple[SingleIndexSelector, SingleColSelector]) -> Any: ...

    @overload
    def __getitem__(  # type: ignore[overload-overlap]
        self, item: str | tuple[MultiIndexSelector, SingleColSelector]
    ) -> Series[Any]: ...

    @overload
    def __getitem__(
        self,
        item: (
            SingleIndexSelector
            | MultiIndexSelector
            | MultiColSelector
            | tuple[SingleIndexSelector, MultiColSelector]
            | tuple[MultiIndexSelector, MultiColSelector]
        ),
    ) -> Self: ...
    def __getitem__(
        self,
        item: (
            SingleIndexSelector
            | SingleColSelector
            | MultiColSelector
            | MultiIndexSelector
            | tuple[SingleIndexSelector, SingleColSelector]
            | tuple[SingleIndexSelector, MultiColSelector]
            | tuple[MultiIndexSelector, SingleColSelector]
            | tuple[MultiIndexSelector, MultiColSelector]
        ),
    ) -> Series[Any] | Self | Any:
        return super().__getitem__(item)

    def lazy(
        self,
        backend: ModuleType | Implementation | str | None = None,
    ) -> LazyFrame[Any]:
        return super().lazy(backend=backend)  # type: ignore[return-value]

    # Not sure what mypy is complaining about, probably some fancy
    # thing that I need to understand category theory for
    @overload  # type: ignore[override]
    def to_dict(self, *, as_series: Literal[True] = ...) -> dict[str, Series[Any]]: ...
    @overload
    def to_dict(self, *, as_series: Literal[False]) -> dict[str, list[Any]]: ...
    @overload
    def to_dict(
        self, *, as_series: bool
    ) -> dict[str, Series[Any]] | dict[str, list[Any]]: ...
    def to_dict(
        self, *, as_series: bool = True
    ) -> dict[str, Series[Any]] | dict[str, list[Any]]:
        return super().to_dict(as_series=as_series)  # type: ignore[return-value]

    def is_duplicated(self) -> Series[Any]:
        return super().is_duplicated()  # type: ignore[return-value]

    def is_unique(self) -> Series[Any]:
        return super().is_unique()  # type: ignore[return-value]

    def _l1_norm(self) -> Self:
        """Private, just used to test the stable API.

        Returns:
            A new DataFrame.
        """
        return self.select(all()._l1_norm())


class LazyFrame(NwLazyFrame[IntoFrameT]):
    @inherit_doc(NwLazyFrame)
    def __init__(self, df: Any, *, level: Literal["full", "lazy", "interchange"]) -> None:
        super().__init__(df, level=level)

    @property
    def _dataframe(self) -> type[DataFrame[Any]]:
        return DataFrame

    def _extract_compliant(self, arg: Any) -> Any:
        # After v1, we raise when passing order-dependent or length-changing
        # expressions to LazyFrame
        from narwhals.dataframe import BaseFrame
        from narwhals.expr import Expr
        from narwhals.series import Series

        if isinstance(arg, BaseFrame):
            return arg._compliant_frame
        if isinstance(arg, Series):  # pragma: no cover
            msg = "Mixing Series with LazyFrame is not supported."
            raise TypeError(msg)
        if isinstance(arg, Expr):
            # After stable.v1, we raise for order-dependent exprs or filtrations
            return arg._to_compliant_expr(self.__narwhals_namespace__())
        if isinstance(arg, str):
            plx = self.__narwhals_namespace__()
            return plx.col(arg)
        if get_polars() is not None and "polars" in str(type(arg)):  # pragma: no cover
            msg = (
                f"Expected Narwhals object, got: {type(arg)}.\n\n"
                "Perhaps you:\n"
                "- Forgot a `nw.from_native` somewhere?\n"
                "- Used `pl.col` instead of `nw.col`?"
            )
            raise TypeError(msg)
        raise InvalidIntoExprError.from_invalid_type(type(arg))

    def collect(
        self,
        backend: ModuleType | Implementation | str | None = None,
        **kwargs: Any,
    ) -> DataFrame[Any]:
        return super().collect(backend=backend, **kwargs)  # type: ignore[return-value]

    def _l1_norm(self) -> Self:
        """Private, just used to test the stable API.

        Returns:
            A new lazyframe.
        """
        return self.select(all()._l1_norm())

    def tail(self, n: int = 5) -> Self:  # pragma: no cover
        r"""Get the last `n` rows.

        Arguments:
            n: Number of rows to return.

        Returns:
            A subset of the LazyFrame of shape (n, n_columns).
        """
        return super().tail(n)

    def gather_every(self, n: int, offset: int = 0) -> Self:
        r"""Take every nth row in the DataFrame and return as a new DataFrame.

        Arguments:
            n: Gather every *n*-th row.
            offset: Starting index.

        Returns:
            The LazyFrame containing only the selected rows.
        """
        return self._with_compliant(
            self._compliant_frame.gather_every(n=n, offset=offset)
        )


class Series(NwSeries[IntoSeriesT]):
    @inherit_doc(NwSeries)
    def __init__(
        self, series: Any, *, level: Literal["full", "lazy", "interchange"]
    ) -> None:
        super().__init__(series, level=level)

    # We need to override any method which don't return Self so that type
    # annotations are correct.

    @property
    def _dataframe(self) -> type[DataFrame[Any]]:
        return DataFrame

    def to_frame(self) -> DataFrame[Any]:
        return super().to_frame()  # type: ignore[return-value]

    def value_counts(
        self,
        *,
        sort: bool = False,
        parallel: bool = False,
        name: str | None = None,
        normalize: bool = False,
    ) -> DataFrame[Any]:
        return super().value_counts(  # type: ignore[return-value]
            sort=sort, parallel=parallel, name=name, normalize=normalize
        )

    def hist(
        self,
        bins: list[float | int] | None = None,
        *,
        bin_count: int | None = None,
        include_breakpoint: bool = True,
    ) -> DataFrame[Any]:
        from narwhals.exceptions import NarwhalsUnstableWarning
        from narwhals.utils import find_stacklevel

        msg = (
            "`Series.hist` is being called from the stable API although considered "
            "an unstable feature."
        )
        warn(message=msg, category=NarwhalsUnstableWarning, stacklevel=find_stacklevel())
        return super().hist(  # type: ignore[return-value]
            bins=bins,
            bin_count=bin_count,
            include_breakpoint=include_breakpoint,
        )


class Expr(NwExpr):
    def _l1_norm(self) -> Self:
        return super()._taxicab_norm()

    def head(self, n: int = 10) -> Self:
        r"""Get the first `n` rows.

        Arguments:
            n: Number of rows to return.

        Returns:
            A new expression.
        """
        return self._with_orderable_filtration(
            lambda plx: self._to_compliant_expr(plx).head(n)
        )

    def tail(self, n: int = 10) -> Self:
        r"""Get the last `n` rows.

        Arguments:
            n: Number of rows to return.

        Returns:
            A new expression.
        """
        return self._with_orderable_filtration(
            lambda plx: self._to_compliant_expr(plx).tail(n)
        )

    def gather_every(self, n: int, offset: int = 0) -> Self:
        r"""Take every nth value in the Series and return as new Series.

        Arguments:
            n: Gather every *n*-th row.
            offset: Starting index.

        Returns:
            A new expression.
        """
        return self._with_orderable_filtration(
            lambda plx: self._to_compliant_expr(plx).gather_every(n=n, offset=offset)
        )

    def unique(self, *, maintain_order: bool | None = None) -> Self:
        """Return unique values of this expression.

        Arguments:
            maintain_order: Keep the same order as the original expression.
                This is deprecated and will be removed in a future version,
                but will still be kept around in `narwhals.stable.v1`.

        Returns:
            A new expression.
        """
        if maintain_order is not None:
            msg = (
                "`maintain_order` has no effect and is only kept around for backwards-compatibility. "
                "You can safely remove this argument."
            )
            warn(message=msg, category=UserWarning, stacklevel=find_stacklevel())
        return self._with_filtration(lambda plx: self._to_compliant_expr(plx).unique())

    def sort(self, *, descending: bool = False, nulls_last: bool = False) -> Self:
        """Sort this column. Place null values first.

        Arguments:
            descending: Sort in descending order.
            nulls_last: Place null values last instead of first.

        Returns:
            A new expression.
        """
        return self._with_unorderable_window(
            lambda plx: self._to_compliant_expr(plx).sort(
                descending=descending, nulls_last=nulls_last
            )
        )

    def arg_true(self) -> Self:
        """Find elements where boolean expression is True.

        Returns:
            A new expression.
        """
        return self._with_orderable_filtration(
            lambda plx: self._to_compliant_expr(plx).arg_true(),
        )

    def sample(
        self,
        n: int | None = None,
        *,
        fraction: float | None = None,
        with_replacement: bool = False,
        seed: int | None = None,
    ) -> Self:
        """Sample randomly from this expression.

        Arguments:
            n: Number of items to return. Cannot be used with fraction.
            fraction: Fraction of items to return. Cannot be used with n.
            with_replacement: Allow values to be sampled more than once.
            seed: Seed for the random number generator. If set to None (default), a random
                seed is generated for each sample operation.

        Returns:
            A new expression.
        """
        return self._with_filtration(
            lambda plx: self._to_compliant_expr(plx).sample(
                n, fraction=fraction, with_replacement=with_replacement, seed=seed
            )
        )


class Schema(NwSchema):
    _version = Version.V1

    @inherit_doc(NwSchema)
    def __init__(
        self, schema: Mapping[str, DType] | Iterable[tuple[str, DType]] | None = None
    ) -> None:
        super().__init__(schema)


@overload
def _stableify(obj: NwDataFrame[IntoFrameT]) -> DataFrame[IntoFrameT]: ...
@overload
def _stableify(obj: NwLazyFrame[IntoFrameT]) -> LazyFrame[IntoFrameT]: ...
@overload
def _stableify(obj: NwSeries[IntoSeriesT]) -> Series[IntoSeriesT]: ...
@overload
def _stableify(obj: NwExpr) -> Expr: ...
@overload
def _stableify(obj: Any) -> Any: ...


def _stableify(
    obj: NwDataFrame[IntoFrameT]
    | NwLazyFrame[IntoFrameT]
    | NwSeries[IntoSeriesT]
    | NwExpr
    | Any,
) -> DataFrame[IntoFrameT] | LazyFrame[IntoFrameT] | Series[IntoSeriesT] | Expr | Any:
    if isinstance(obj, NwDataFrame):
        return DataFrame(obj._compliant_frame._with_version(Version.V1), level=obj._level)
    if isinstance(obj, NwLazyFrame):
        return LazyFrame(obj._compliant_frame._with_version(Version.V1), level=obj._level)
    if isinstance(obj, NwSeries):
        return Series(obj._compliant_series._with_version(Version.V1), level=obj._level)
    if isinstance(obj, NwExpr):
        return Expr(obj._to_compliant_expr, obj._metadata)
    return obj


@overload
def from_native(native_object: SeriesT, **kwds: Any) -> SeriesT: ...


@overload
def from_native(native_object: DataFrameT, **kwds: Any) -> DataFrameT: ...


@overload
def from_native(native_object: LazyFrameT, **kwds: Any) -> LazyFrameT: ...


@overload
def from_native(
    native_object: DataFrameT | LazyFrameT, **kwds: Any
) -> DataFrameT | LazyFrameT: ...


@overload
def from_native(
    native_object: IntoDataFrameT | IntoSeriesT,
    *,
    strict: Literal[False],
    eager_only: Literal[False] = ...,
    eager_or_interchange_only: Literal[True],
    series_only: Literal[False] = ...,
    allow_series: Literal[True],
) -> DataFrame[IntoDataFrameT] | Series[IntoSeriesT]: ...


@overload
def from_native(
    native_object: IntoDataFrameT | IntoSeriesT,
    *,
    strict: Literal[False],
    eager_only: Literal[True],
    eager_or_interchange_only: Literal[False] = ...,
    series_only: Literal[False] = ...,
    allow_series: Literal[True],
) -> DataFrame[IntoDataFrameT] | Series[IntoSeriesT]: ...


@overload
def from_native(
    native_object: IntoDataFrameT,
    *,
    strict: Literal[False],
    eager_only: Literal[False] = ...,
    eager_or_interchange_only: Literal[True],
    series_only: Literal[False] = ...,
    allow_series: None = ...,
) -> DataFrame[IntoDataFrameT]: ...


@overload
def from_native(
    native_object: T,
    *,
    strict: Literal[False],
    eager_only: Literal[False] = ...,
    eager_or_interchange_only: Literal[True],
    series_only: Literal[False] = ...,
    allow_series: None = ...,
) -> T: ...


@overload
def from_native(
    native_object: IntoDataFrameT,
    *,
    strict: Literal[False],
    eager_only: Literal[True],
    eager_or_interchange_only: Literal[False] = ...,
    series_only: Literal[False] = ...,
    allow_series: None = ...,
) -> DataFrame[IntoDataFrameT]: ...


@overload
def from_native(
    native_object: T,
    *,
    strict: Literal[False],
    eager_only: Literal[True],
    eager_or_interchange_only: Literal[False] = ...,
    series_only: Literal[False] = ...,
    allow_series: None = ...,
) -> T: ...


@overload
def from_native(
    native_object: IntoFrameT | IntoSeriesT,
    *,
    strict: Literal[False],
    eager_only: Literal[False] = ...,
    eager_or_interchange_only: Literal[False] = ...,
    series_only: Literal[False] = ...,
    allow_series: Literal[True],
) -> DataFrame[IntoFrameT] | LazyFrame[IntoFrameT] | Series[IntoSeriesT]: ...


@overload
def from_native(
    native_object: IntoSeriesT,
    *,
    strict: Literal[False],
    eager_only: Literal[False] = ...,
    eager_or_interchange_only: Literal[False] = ...,
    series_only: Literal[True],
    allow_series: None = ...,
) -> Series[IntoSeriesT]: ...


@overload
def from_native(
    native_object: IntoFrameT,
    *,
    strict: Literal[False],
    eager_only: Literal[False] = ...,
    eager_or_interchange_only: Literal[False] = ...,
    series_only: Literal[False] = ...,
    allow_series: None = ...,
) -> DataFrame[IntoFrameT] | LazyFrame[IntoFrameT]: ...


@overload
def from_native(
    native_object: T,
    *,
    strict: Literal[False],
    eager_only: Literal[False] = ...,
    eager_or_interchange_only: Literal[False] = ...,
    series_only: Literal[False] = ...,
    allow_series: None = ...,
) -> T: ...


@overload
def from_native(
    native_object: IntoDataFrameT,
    *,
    strict: Literal[True] = ...,
    eager_only: Literal[False] = ...,
    eager_or_interchange_only: Literal[True],
    series_only: Literal[False] = ...,
    allow_series: None = ...,
) -> DataFrame[IntoDataFrameT]: ...


@overload
def from_native(
    native_object: IntoDataFrameT,
    *,
    strict: Literal[True] = ...,
    eager_only: Literal[True],
    eager_or_interchange_only: Literal[False] = ...,
    series_only: Literal[False] = ...,
    allow_series: None = ...,
) -> DataFrame[IntoDataFrameT]: ...


@overload
def from_native(
    native_object: IntoFrame | IntoSeries,
    *,
    strict: Literal[True] = ...,
    eager_only: Literal[False] = ...,
    eager_or_interchange_only: Literal[False] = ...,
    series_only: Literal[False] = ...,
    allow_series: Literal[True],
) -> DataFrame[Any] | LazyFrame[Any] | Series[Any]: ...


@overload
def from_native(
    native_object: IntoSeriesT,
    *,
    strict: Literal[True] = ...,
    eager_only: Literal[False] = ...,
    eager_or_interchange_only: Literal[False] = ...,
    series_only: Literal[True],
    allow_series: None = ...,
) -> Series[IntoSeriesT]: ...


@overload
def from_native(
    native_object: IntoLazyFrameT,
    *,
    strict: Literal[True] = ...,
    eager_only: Literal[False] = ...,
    eager_or_interchange_only: Literal[False] = ...,
    series_only: Literal[False] = ...,
    allow_series: None = ...,
) -> LazyFrame[IntoLazyFrameT]: ...


# NOTE: `pl.LazyFrame` originally matched here
@overload
def from_native(
    native_object: IntoFrameT,
    *,
    strict: Literal[True] = ...,
    eager_only: Literal[False] = ...,
    eager_or_interchange_only: Literal[False] = ...,
    series_only: Literal[False] = ...,
    allow_series: None = ...,
) -> DataFrame[IntoFrameT] | LazyFrame[IntoFrameT]: ...


@overload
def from_native(
    native_object: IntoDataFrameT | IntoSeries,
    *,
    pass_through: Literal[True],
    eager_only: Literal[False] = ...,
    eager_or_interchange_only: Literal[True],
    series_only: Literal[False] = ...,
    allow_series: Literal[True],
) -> DataFrame[IntoDataFrameT]: ...


@overload
def from_native(
    native_object: IntoDataFrameT | IntoSeriesT,
    *,
    pass_through: Literal[True],
    eager_only: Literal[True],
    eager_or_interchange_only: Literal[False] = ...,
    series_only: Literal[False] = ...,
    allow_series: Literal[True],
) -> DataFrame[IntoDataFrameT] | Series[IntoSeriesT]: ...


@overload
def from_native(
    native_object: IntoDataFrameT,
    *,
    pass_through: Literal[True],
    eager_only: Literal[False] = ...,
    eager_or_interchange_only: Literal[True],
    series_only: Literal[False] = ...,
    allow_series: None = ...,
) -> DataFrame[IntoDataFrameT]: ...


@overload
def from_native(
    native_object: T,
    *,
    pass_through: Literal[True],
    eager_only: Literal[False] = ...,
    eager_or_interchange_only: Literal[True],
    series_only: Literal[False] = ...,
    allow_series: None = ...,
) -> T: ...


@overload
def from_native(
    native_object: IntoDataFrameT,
    *,
    pass_through: Literal[True],
    eager_only: Literal[True],
    eager_or_interchange_only: Literal[False] = ...,
    series_only: Literal[False] = ...,
    allow_series: None = ...,
) -> DataFrame[IntoDataFrameT]: ...


@overload
def from_native(
    native_object: T,
    *,
    pass_through: Literal[True],
    eager_only: Literal[True],
    eager_or_interchange_only: Literal[False] = ...,
    series_only: Literal[False] = ...,
    allow_series: None = ...,
) -> T: ...


@overload
def from_native(
    native_object: IntoFrameT | IntoSeriesT,
    *,
    pass_through: Literal[True],
    eager_only: Literal[False] = ...,
    eager_or_interchange_only: Literal[False] = ...,
    series_only: Literal[False] = ...,
    allow_series: Literal[True],
) -> DataFrame[IntoFrameT] | LazyFrame[IntoFrameT] | Series[IntoSeriesT]: ...


@overload
def from_native(
    native_object: IntoSeriesT,
    *,
    pass_through: Literal[True],
    eager_only: Literal[False] = ...,
    eager_or_interchange_only: Literal[False] = ...,
    series_only: Literal[True],
    allow_series: None = ...,
) -> Series[IntoSeriesT]: ...


@overload
def from_native(
    native_object: IntoFrameT,
    *,
    pass_through: Literal[True],
    eager_only: Literal[False] = ...,
    eager_or_interchange_only: Literal[False] = ...,
    series_only: Literal[False] = ...,
    allow_series: None = ...,
) -> DataFrame[IntoFrameT] | LazyFrame[IntoFrameT]: ...


@overload
def from_native(
    native_object: T,
    *,
    pass_through: Literal[True],
    eager_only: Literal[False] = ...,
    eager_or_interchange_only: Literal[False] = ...,
    series_only: Literal[False] = ...,
    allow_series: None = ...,
) -> T: ...


@overload
def from_native(
    native_object: IntoDataFrameT,
    *,
    pass_through: Literal[False] = ...,
    eager_only: Literal[False] = ...,
    eager_or_interchange_only: Literal[True],
    series_only: Literal[False] = ...,
    allow_series: None = ...,
) -> DataFrame[IntoDataFrameT]: ...


@overload
def from_native(
    native_object: IntoDataFrameT,
    *,
    pass_through: Literal[False] = ...,
    eager_only: Literal[True],
    eager_or_interchange_only: Literal[False] = ...,
    series_only: Literal[False] = ...,
    allow_series: None = ...,
) -> DataFrame[IntoDataFrameT]: ...


@overload
def from_native(
    native_object: IntoFrame | IntoSeries,
    *,
    pass_through: Literal[False] = ...,
    eager_only: Literal[False] = ...,
    eager_or_interchange_only: Literal[False] = ...,
    series_only: Literal[False] = ...,
    allow_series: Literal[True],
) -> DataFrame[Any] | LazyFrame[Any] | Series[Any]: ...


@overload
def from_native(
    native_object: IntoSeriesT,
    *,
    pass_through: Literal[False] = ...,
    eager_only: Literal[False] = ...,
    eager_or_interchange_only: Literal[False] = ...,
    series_only: Literal[True],
    allow_series: None = ...,
) -> Series[IntoSeriesT]: ...


@overload
def from_native(
    native_object: IntoFrameT,
    *,
    pass_through: Literal[False] = ...,
    eager_only: Literal[False] = ...,
    eager_or_interchange_only: Literal[False] = ...,
    series_only: Literal[False] = ...,
    allow_series: None = ...,
) -> DataFrame[IntoFrameT] | LazyFrame[IntoFrameT]: ...


# All params passed in as variables
@overload
def from_native(
    native_object: Any,
    *,
    pass_through: bool,
    eager_only: bool,
    eager_or_interchange_only: bool = False,
    series_only: bool,
    allow_series: bool | None,
) -> Any: ...


def from_native(  # noqa: D417
    native_object: IntoFrameT | IntoFrame | IntoSeriesT | IntoSeries | T,
    *,
    strict: bool | None = None,
    pass_through: bool | None = None,
    eager_only: bool = False,
    eager_or_interchange_only: bool = False,
    series_only: bool = False,
    allow_series: bool | None = None,
    **kwds: Any,
) -> LazyFrame[IntoFrameT] | DataFrame[IntoFrameT] | Series[IntoSeriesT] | T:
    """Convert `native_object` to Narwhals Dataframe, Lazyframe, or Series.

    Arguments:
        native_object: Raw object from user.
            Depending on the other arguments, input object can be

            - a Dataframe / Lazyframe / Series supported by Narwhals (pandas, Polars, PyArrow, ...)
            - an object which implements `__narwhals_dataframe__`, `__narwhals_lazyframe__`,
              or `__narwhals_series__`
        strict: Determine what happens if the object can't be converted to Narwhals

            - `True` or `None` (default): raise an error
            - `False`: pass object through as-is

            *Deprecated* (v1.13.0)

            Please use `pass_through` instead. Note that `strict` is still available
            (and won't emit a deprecation warning) if you use `narwhals.stable.v1`,
            see [perfect backwards compatibility policy](../backcompat.md/).
        pass_through: Determine what happens if the object can't be converted to Narwhals

            - `False` or `None` (default): raise an error
            - `True`: pass object through as-is
        eager_only: Whether to only allow eager objects

            - `False` (default): don't require `native_object` to be eager
            - `True`: only convert to Narwhals if `native_object` is eager
        eager_or_interchange_only: Whether to only allow eager objects or objects which
            have interchange-level support in Narwhals

            - `False` (default): don't require `native_object` to either be eager or to
              have interchange-level support in Narwhals
            - `True`: only convert to Narwhals if `native_object` is eager or has
              interchange-level support in Narwhals

            See [interchange-only support](../extending.md/#interchange-only-support)
            for more details.
        series_only: Whether to only allow Series

            - `False` (default): don't require `native_object` to be a Series
            - `True`: only convert to Narwhals if `native_object` is a Series
        allow_series: Whether to allow Series (default is only Dataframe / Lazyframe)

            - `False` or `None` (default): don't convert to Narwhals if `native_object` is a Series
            - `True`: allow `native_object` to be a Series

    Returns:
        DataFrame, LazyFrame, Series, or original object, depending
            on which combination of parameters was passed.
    """
    # Early returns
    if isinstance(native_object, (DataFrame, LazyFrame)) and not series_only:
        return native_object
    if isinstance(native_object, Series) and (series_only or allow_series):
        return native_object

    pass_through = validate_strict_and_pass_though(
        strict, pass_through, pass_through_default=False, emit_deprecation_warning=False
    )
    if kwds:
        msg = f"from_native() got an unexpected keyword argument {next(iter(kwds))!r}"
        raise TypeError(msg)

    result = _from_native_impl(
        native_object,
        pass_through=pass_through,
        eager_only=eager_only,
        eager_or_interchange_only=eager_or_interchange_only,
        series_only=series_only,
        allow_series=allow_series,
        version=Version.V1,
    )
    return _stableify(result)  # type: ignore[no-any-return]


@overload
def to_native(
    narwhals_object: DataFrame[IntoDataFrameT], *, strict: Literal[True] = ...
) -> IntoDataFrameT: ...
@overload
def to_native(
    narwhals_object: LazyFrame[IntoFrameT], *, strict: Literal[True] = ...
) -> IntoFrameT: ...
@overload
def to_native(
    narwhals_object: Series[IntoSeriesT], *, strict: Literal[True] = ...
) -> IntoSeriesT: ...
@overload
def to_native(narwhals_object: Any, *, strict: bool) -> Any: ...
@overload
def to_native(
    narwhals_object: DataFrame[IntoDataFrameT], *, pass_through: Literal[False] = ...
) -> IntoDataFrameT: ...
@overload
def to_native(
    narwhals_object: LazyFrame[IntoFrameT], *, pass_through: Literal[False] = ...
) -> IntoFrameT: ...
@overload
def to_native(
    narwhals_object: Series[IntoSeriesT], *, pass_through: Literal[False] = ...
) -> IntoSeriesT: ...
@overload
def to_native(narwhals_object: Any, *, pass_through: bool) -> Any: ...


def to_native(
    narwhals_object: DataFrame[IntoDataFrameT]
    | LazyFrame[IntoFrameT]
    | Series[IntoSeriesT],
    *,
    strict: bool | None = None,
    pass_through: bool | None = None,
) -> IntoFrameT | IntoSeriesT | Any:
    """Convert Narwhals object to native one.

    Arguments:
        narwhals_object: Narwhals object.
        strict: Determine what happens if `narwhals_object` isn't a Narwhals class

            - `True` (default): raise an error
            - `False`: pass object through as-is

            *Deprecated* (v1.13.0)

            Please use `pass_through` instead. Note that `strict` is still available
            (and won't emit a deprecation warning) if you use `narwhals.stable.v1`,
            see [perfect backwards compatibility policy](../backcompat.md/).
        pass_through: Determine what happens if `narwhals_object` isn't a Narwhals class

            - `False` (default): raise an error
            - `True`: pass object through as-is

    Returns:
        Object of class that user started with.
    """
    from narwhals.dataframe import BaseFrame
    from narwhals.series import Series
    from narwhals.utils import validate_strict_and_pass_though

    pass_through = validate_strict_and_pass_though(
        strict, pass_through, pass_through_default=False, emit_deprecation_warning=False
    )

    if isinstance(narwhals_object, BaseFrame):
        return narwhals_object._compliant_frame._native_frame
    if isinstance(narwhals_object, Series):
        return narwhals_object._compliant_series.native

    if not pass_through:
        msg = f"Expected Narwhals object, got {type(narwhals_object)}."
        raise TypeError(msg)
    return narwhals_object


def narwhalify(
    func: Callable[..., Any] | None = None,
    *,
    strict: bool | None = None,
    pass_through: bool | None = None,
    eager_only: bool = False,
    eager_or_interchange_only: bool = False,
    series_only: bool = False,
    allow_series: bool | None = True,
) -> Callable[..., Any]:
    """Decorate function so it becomes dataframe-agnostic.

    This will try to convert any dataframe/series-like object into the Narwhals
    respective DataFrame/Series, while leaving the other parameters as they are.
    Similarly, if the output of the function is a Narwhals DataFrame or Series, it will be
    converted back to the original dataframe/series type, while if the output is another
    type it will be left as is.
    By setting `pass_through=False`, then every input and every output will be required to be a
    dataframe/series-like object.

    Arguments:
        func: Function to wrap in a `from_native`-`to_native` block.
        strict: Determine what happens if the object can't be converted to Narwhals

            *Deprecated* (v1.13.0)

            Please use `pass_through` instead. Note that `strict` is still available
            (and won't emit a deprecation warning) if you use `narwhals.stable.v1`,
            see [perfect backwards compatibility policy](../backcompat.md/).

            - `True` or `None` (default): raise an error
            - `False`: pass object through as-is
        pass_through: Determine what happens if the object can't be converted to Narwhals

            - `False` or `None` (default): raise an error
            - `True`: pass object through as-is
        eager_only: Whether to only allow eager objects

            - `False` (default): don't require `native_object` to be eager
            - `True`: only convert to Narwhals if `native_object` is eager
        eager_or_interchange_only: Whether to only allow eager objects or objects which
            have interchange-level support in Narwhals

            - `False` (default): don't require `native_object` to either be eager or to
              have interchange-level support in Narwhals
            - `True`: only convert to Narwhals if `native_object` is eager or has
              interchange-level support in Narwhals

            See [interchange-only support](../extending.md/#interchange-only-support)
            for more details.
        series_only: Whether to only allow Series

            - `False` (default): don't require `native_object` to be a Series
            - `True`: only convert to Narwhals if `native_object` is a Series
        allow_series: Whether to allow Series (default is only Dataframe / Lazyframe)

            - `False` or `None`: don't convert to Narwhals if `native_object` is a Series
            - `True` (default): allow `native_object` to be a Series

    Returns:
        Decorated function.
    """
    pass_through = validate_strict_and_pass_though(
        strict, pass_through, pass_through_default=True, emit_deprecation_warning=False
    )

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            args = [
                from_native(
                    arg,
                    pass_through=pass_through,
                    eager_only=eager_only,
                    eager_or_interchange_only=eager_or_interchange_only,
                    series_only=series_only,
                    allow_series=allow_series,
                )
                for arg in args
            ]  # type: ignore[assignment]

            kwargs = {
                name: from_native(
                    value,
                    pass_through=pass_through,
                    eager_only=eager_only,
                    eager_or_interchange_only=eager_or_interchange_only,
                    series_only=series_only,
                    allow_series=allow_series,
                )
                for name, value in kwargs.items()
            }

            backends = {
                b()
                for v in (*args, *kwargs.values())
                if (b := getattr(v, "__native_namespace__", None))
            }

            if backends.__len__() > 1:
                msg = "Found multiple backends. Make sure that all dataframe/series inputs come from the same backend."
                raise ValueError(msg)

            result = func(*args, **kwargs)

            return to_native(result, pass_through=pass_through)

        return wrapper

    if func is None:
        return decorator
    else:
        # If func is not None, it means the decorator is used without arguments
        return decorator(func)


def all() -> Expr:
    """Instantiate an expression representing all columns.

    Returns:
        A new expression.
    """
    return _stableify(nw.all())


def col(*names: str | Iterable[str]) -> Expr:
    """Creates an expression that references one or more columns by their name(s).

    Arguments:
        names: Name(s) of the columns to use.

    Returns:
        A new expression.
    """
    return _stableify(nw.col(*names))


def exclude(*names: str | Iterable[str]) -> Expr:
    """Creates an expression that excludes columns by their name(s).

    Arguments:
        names: Name(s) of the columns to exclude.

    Returns:
        A new expression.
    """
    return _stableify(nw.exclude(*names))


def nth(*indices: int | Sequence[int]) -> Expr:
    """Creates an expression that references one or more columns by their index(es).

    Notes:
        `nth` is not supported for Polars version<1.0.0. Please use
        [`narwhals.col`][] instead.

    Arguments:
        indices: One or more indices representing the columns to retrieve.

    Returns:
        A new expression.
    """
    return _stableify(nw.nth(*indices))


def len() -> Expr:
    """Return the number of rows.

    Returns:
        A new expression.
    """
    return _stableify(nw.len())


def lit(value: NonNestedLiteral, dtype: DType | type[DType] | None = None) -> Expr:
    """Return an expression representing a literal value.

    Arguments:
        value: The value to use as literal.
        dtype: The data type of the literal value. If not provided, the data type will
            be inferred by the native library.

    Returns:
        A new expression.
    """
    return _stableify(nw.lit(value, dtype))


def min(*columns: str) -> Expr:
    """Return the minimum value.

    Note:
       Syntactic sugar for ``nw.col(columns).min()``.

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function.

    Returns:
        A new expression.
    """
    return _stableify(nw.min(*columns))


def max(*columns: str) -> Expr:
    """Return the maximum value.

    Note:
       Syntactic sugar for ``nw.col(columns).max()``.

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function.

    Returns:
        A new expression.
    """
    return _stableify(nw.max(*columns))


def mean(*columns: str) -> Expr:
    """Get the mean value.

    Note:
        Syntactic sugar for ``nw.col(columns).mean()``

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function

    Returns:
        A new expression.
    """
    return _stableify(nw.mean(*columns))


def median(*columns: str) -> Expr:
    """Get the median value.

    Notes:
        - Syntactic sugar for ``nw.col(columns).median()``
        - Results might slightly differ across backends due to differences in the
            underlying algorithms used to compute the median.

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function

    Returns:
        A new expression.
    """
    return _stableify(nw.median(*columns))


def sum(*columns: str) -> Expr:
    """Sum all values.

    Note:
        Syntactic sugar for ``nw.col(columns).sum()``

    Arguments:
        columns: Name(s) of the columns to use in the aggregation function

    Returns:
        A new expression.
    """
    return _stableify(nw.sum(*columns))


def sum_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """Sum all values horizontally across columns.

    Warning:
        Unlike Polars, we support horizontal sum over numeric columns only.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Returns:
        A new expression.
    """
    return _stableify(nw.sum_horizontal(*exprs))


def all_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    r"""Compute the bitwise AND horizontally across columns.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Returns:
        A new expression.
    """
    return _stableify(nw.all_horizontal(*exprs))


def any_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    r"""Compute the bitwise OR horizontally across columns.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Returns:
        A new expression.
    """
    return _stableify(nw.any_horizontal(*exprs))


def mean_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """Compute the mean of all values horizontally across columns.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Returns:
        A new expression.
    """
    return _stableify(nw.mean_horizontal(*exprs))


def min_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """Get the minimum value horizontally across columns.

    Notes:
        We support `min_horizontal` over numeric columns only.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Returns:
        A new expression.
    """
    return _stableify(nw.min_horizontal(*exprs))


def max_horizontal(*exprs: IntoExpr | Iterable[IntoExpr]) -> Expr:
    """Get the maximum value horizontally across columns.

    Notes:
        We support `max_horizontal` over numeric columns only.

    Arguments:
        exprs: Name(s) of the columns to use in the aggregation function. Accepts
            expression input.

    Returns:
        A new expression.
    """
    return _stableify(nw.max_horizontal(*exprs))


def concat(items: Iterable[FrameT], *, how: ConcatMethod = "vertical") -> FrameT:
    """Concatenate multiple DataFrames, LazyFrames into a single entity.

    Arguments:
        items: DataFrames, LazyFrames to concatenate.
        how: concatenating strategy

            - vertical: Concatenate vertically. Column names must match.
            - horizontal: Concatenate horizontally. If lengths don't match, then
                missing rows are filled with null values. This is only supported
                when all inputs are (eager) DataFrames.
            - diagonal: Finds a union between the column schemas and fills missing column
                values with null.

    Returns:
        A new DataFrame or LazyFrame resulting from the concatenation.

    Raises:
        TypeError: The items to concatenate should either all be eager, or all lazy
    """
    return cast("FrameT", _stableify(nw.concat(items, how=how)))


def concat_str(
    exprs: IntoExpr | Iterable[IntoExpr],
    *more_exprs: IntoExpr,
    separator: str = "",
    ignore_nulls: bool = False,
) -> Expr:
    r"""Horizontally concatenate columns into a single string column.

    Arguments:
        exprs: Columns to concatenate into a single string column. Accepts expression
            input. Strings are parsed as column names, other non-expression inputs are
            parsed as literals. Non-`String` columns are cast to `String`.
        *more_exprs: Additional columns to concatenate into a single string column,
            specified as positional arguments.
        separator: String that will be used to separate the values of each column.
        ignore_nulls: Ignore null values (default is `False`).
            If set to `False`, null values will be propagated and if the row contains any
            null values, the output is null.

    Returns:
        A new expression.
    """
    return _stableify(
        nw.concat_str(exprs, *more_exprs, separator=separator, ignore_nulls=ignore_nulls)
    )


class When(NwWhen):
    @classmethod
    def from_when(cls, when: NwWhen) -> When:
        return cls(when._predicate)

    def then(self, value: IntoExpr | NonNestedLiteral | _1DArray) -> Then:
        return Then.from_then(super().then(value))


class Then(NwThen, Expr):
    @classmethod
    def from_then(cls, then: NwThen) -> Then:
        return cls(then._to_compliant_expr, then._metadata)

    def otherwise(self, value: IntoExpr | NonNestedLiteral | _1DArray) -> Expr:
        return _stableify(super().otherwise(value))


def when(*predicates: IntoExpr | Iterable[IntoExpr]) -> When:
    """Start a `when-then-otherwise` expression.

    Expression similar to an `if-else` statement in Python. Always initiated by a
    `pl.when(<condition>).then(<value if condition>)`, and optionally followed by a
    `.otherwise(<value if condition is false>)` can be appended at the end. If not
    appended, and the condition is not `True`, `None` will be returned.

    Info:
        Chaining multiple `.when(<condition>).then(<value>)` statements is currently
        not supported.
        See [Narwhals#668](https://github.com/narwhals-dev/narwhals/issues/668).

    Arguments:
        predicates: Condition(s) that must be met in order to apply the subsequent
            statement. Accepts one or more boolean expressions, which are implicitly
            combined with `&`. String input is parsed as a column name.

    Returns:
        A "when" object, which `.then` can be called on.
    """
    return When.from_when(nw_when(*predicates))


@deprecate_native_namespace(required=True)
def new_series(
    name: str,
    values: Any,
    dtype: DType | type[DType] | None = None,
    *,
    backend: ModuleType | Implementation | str | None = None,
    native_namespace: ModuleType | None = None,  # noqa: ARG001
) -> Series[Any]:
    """Instantiate Narwhals Series from iterable (e.g. list or array).

    Arguments:
        name: Name of resulting Series.
        values: Values of make Series from.
        dtype: (Narwhals) dtype. If not provided, the native library
            may auto-infer it from `values`.
        backend: specifies which eager backend instantiate to.

            `backend` can be specified in various ways

            - As `Implementation.<BACKEND>` with `BACKEND` being `PANDAS`, `PYARROW`,
                `POLARS`, `MODIN` or `CUDF`.
            - As a string: `"pandas"`, `"pyarrow"`, `"polars"`, `"modin"` or `"cudf"`.
            - Directly as a module `pandas`, `pyarrow`, `polars`, `modin` or `cudf`.
        native_namespace: The native library to use for DataFrame creation.

            *Deprecated* (v1.31.0)

            Please use `backend` instead. Note that `native_namespace` is still available
            (and won't emit a deprecation warning) if you use `narwhals.stable.v1`,
            see [perfect backwards compatibility policy](../backcompat.md/).

    Returns:
        A new Series
    """
    backend = cast("ModuleType | Implementation | str", backend)
    return _stableify(  # type: ignore[no-any-return]
        _new_series_impl(name, values, dtype, backend=backend, version=Version.V1)
    )


@deprecate_native_namespace(required=True)
def from_arrow(
    native_frame: IntoArrowTable,
    *,
    backend: ModuleType | Implementation | str | None = None,
    native_namespace: ModuleType | None = None,  # noqa: ARG001
) -> DataFrame[Any]:
    """Construct a DataFrame from an object which supports the PyCapsule Interface.

    Arguments:
        native_frame: Object which implements `__arrow_c_stream__`.
        backend: specifies which eager backend instantiate to.

            `backend` can be specified in various ways

            - As `Implementation.<BACKEND>` with `BACKEND` being `PANDAS`, `PYARROW`,
                `POLARS`, `MODIN` or `CUDF`.
            - As a string: `"pandas"`, `"pyarrow"`, `"polars"`, `"modin"` or `"cudf"`.
            - Directly as a module `pandas`, `pyarrow`, `polars`, `modin` or `cudf`.
        native_namespace: The native library to use for DataFrame creation.

            *Deprecated* (v1.31.0)

            Please use `backend` instead. Note that `native_namespace` is still available
            (and won't emit a deprecation warning) if you use `narwhals.stable.v1`,
            see [perfect backwards compatibility policy](../backcompat.md/).

    Returns:
        A new DataFrame.
    """
    backend = cast("ModuleType | Implementation | str", backend)
    return _stableify(  # type: ignore[no-any-return]
        _from_arrow_impl(native_frame, backend=backend, version=Version.V1)
    )


@deprecate_native_namespace()
def from_dict(
    data: Mapping[str, Any],
    schema: Mapping[str, DType] | Schema | None = None,
    *,
    backend: ModuleType | Implementation | str | None = None,
    native_namespace: ModuleType | None = None,  # noqa: ARG001
) -> DataFrame[Any]:
    """Instantiate DataFrame from dictionary.

    Indexes (if present, for pandas-like backends) are aligned following
    the [left-hand-rule](../concepts/pandas_index.md/).

    Notes:
        For pandas-like dataframes, conversion to schema is applied after dataframe
        creation.

    Arguments:
        data: Dictionary to create DataFrame from.
        schema: The DataFrame schema as Schema or dict of {name: type}. If not
            specified, the schema will be inferred by the native library.
        backend: specifies which eager backend instantiate to. Only
            necessary if inputs are not Narwhals Series.

            `backend` can be specified in various ways

            - As `Implementation.<BACKEND>` with `BACKEND` being `PANDAS`, `PYARROW`,
                `POLARS`, `MODIN` or `CUDF`.
            - As a string: `"pandas"`, `"pyarrow"`, `"polars"`, `"modin"` or `"cudf"`.
            - Directly as a module `pandas`, `pyarrow`, `polars`, `modin` or `cudf`.
        native_namespace: The native library to use for DataFrame creation.

            *Deprecated* (v1.26.0)

            Please use `backend` instead. Note that `native_namespace` is still available
            (and won't emit a deprecation warning) if you use `narwhals.stable.v1`,
            see [perfect backwards compatibility policy](../backcompat.md/).

    Returns:
        A new DataFrame.
    """
    return _stableify(  # type: ignore[no-any-return]
        _from_dict_impl(data, schema, backend=backend, version=Version.V1)
    )


@deprecate_native_namespace(required=True)
def from_numpy(
    data: _2DArray,
    schema: Mapping[str, DType] | Schema | Sequence[str] | None = None,
    *,
    backend: ModuleType | Implementation | str | None = None,
    native_namespace: ModuleType | None = None,  # noqa: ARG001
) -> DataFrame[Any]:
    """Construct a DataFrame from a NumPy ndarray.

    Notes:
        Only row orientation is currently supported.

        For pandas-like dataframes, conversion to schema is applied after dataframe
        creation.

    Arguments:
        data: Two-dimensional data represented as a NumPy ndarray.
        schema: The DataFrame schema as Schema, dict of {name: type}, or a sequence of str.
        backend: specifies which eager backend instantiate to.

            `backend` can be specified in various ways

            - As `Implementation.<BACKEND>` with `BACKEND` being `PANDAS`, `PYARROW`,
                `POLARS`, `MODIN` or `CUDF`.
            - As a string: `"pandas"`, `"pyarrow"`, `"polars"`, `"modin"` or `"cudf"`.
            - Directly as a module `pandas`, `pyarrow`, `polars`, `modin` or `cudf`.
        native_namespace: The native library to use for DataFrame creation.

            *Deprecated* (v1.31.0)

            Please use `backend` instead. Note that `native_namespace` is still available
            (and won't emit a deprecation warning) if you use `narwhals.stable.v1`,
            see [perfect backwards compatibility policy](../backcompat.md/).

    Returns:
        A new DataFrame.
    """
    backend = cast("ModuleType | Implementation | str", backend)
    return _stableify(_from_numpy_impl(data, schema, backend=backend, version=Version.V1))  # type: ignore[no-any-return]


@deprecate_native_namespace(required=True)
def read_csv(
    source: str,
    *,
    backend: ModuleType | Implementation | str | None = None,
    native_namespace: ModuleType | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> DataFrame[Any]:
    """Read a CSV file into a DataFrame.

    Arguments:
        source: Path to a file.
        backend: The eager backend for DataFrame creation.
            `backend` can be specified in various ways

            - As `Implementation.<BACKEND>` with `BACKEND` being `PANDAS`, `PYARROW`,
                `POLARS`, `MODIN` or `CUDF`.
            - As a string: `"pandas"`, `"pyarrow"`, `"polars"`, `"modin"` or `"cudf"`.
            - Directly as a module `pandas`, `pyarrow`, `polars`, `modin` or `cudf`.
        native_namespace: The native library to use for DataFrame creation.

            *Deprecated* (v1.27.2)

            Please use `backend` instead. Note that `native_namespace` is still available
            (and won't emit a deprecation warning) if you use `narwhals.stable.v1`,
            see [perfect backwards compatibility policy](../backcompat.md/).
        kwargs: Extra keyword arguments which are passed to the native CSV reader.
            For example, you could use
            `nw.read_csv('file.csv', backend='pandas', engine='pyarrow')`.

    Returns:
        DataFrame.
    """
    backend = cast("ModuleType | Implementation | str", backend)
    return _stableify(  # type: ignore[no-any-return]
        _read_csv_impl(source, backend=backend, **kwargs)
    )


@deprecate_native_namespace(required=True)
def scan_csv(
    source: str,
    *,
    backend: ModuleType | Implementation | str | None = None,
    native_namespace: ModuleType | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> LazyFrame[Any]:
    """Lazily read from a CSV file.

    For the libraries that do not support lazy dataframes, the function reads
    a csv file eagerly and then converts the resulting dataframe to a lazyframe.

    Arguments:
        source: Path to a file.
        backend: The eager backend for DataFrame creation.
            `backend` can be specified in various ways

            - As `Implementation.<BACKEND>` with `BACKEND` being `PANDAS`, `PYARROW`,
                `POLARS`, `MODIN` or `CUDF`.
            - As a string: `"pandas"`, `"pyarrow"`, `"polars"`, `"modin"` or `"cudf"`.
            - Directly as a module `pandas`, `pyarrow`, `polars`, `modin` or `cudf`.
        native_namespace: The native library to use for DataFrame creation.

            *Deprecated* (v1.31.0)

            Please use `backend` instead. Note that `native_namespace` is still available
            (and won't emit a deprecation warning) if you use `narwhals.stable.v1`,
            see [perfect backwards compatibility policy](../backcompat.md/).
        kwargs: Extra keyword arguments which are passed to the native CSV reader.
            For example, you could use
            `nw.scan_csv('file.csv', backend=pd, engine='pyarrow')`.

    Returns:
        LazyFrame.
    """
    backend = cast("ModuleType | Implementation | str", backend)
    return _stableify(  # type: ignore[no-any-return]
        _scan_csv_impl(source, backend=backend, **kwargs)
    )


@deprecate_native_namespace(required=True)
def read_parquet(
    source: str,
    *,
    backend: ModuleType | Implementation | str | None = None,
    native_namespace: ModuleType | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> DataFrame[Any]:
    """Read into a DataFrame from a parquet file.

    Arguments:
        source: Path to a file.
        backend: The eager backend for DataFrame creation.
            `backend` can be specified in various ways

            - As `Implementation.<BACKEND>` with `BACKEND` being `PANDAS`, `PYARROW`,
                `POLARS`, `MODIN` or `CUDF`.
            - As a string: `"pandas"`, `"pyarrow"`, `"polars"`, `"modin"` or `"cudf"`.
            - Directly as a module `pandas`, `pyarrow`, `polars`, `modin` or `cudf`.
        native_namespace: The native library to use for DataFrame creation.

            *Deprecated* (v1.31.0)

            Please use `backend` instead. Note that `native_namespace` is still available
            (and won't emit a deprecation warning) if you use `narwhals.stable.v1`,
            see [perfect backwards compatibility policy](../backcompat.md/).
        kwargs: Extra keyword arguments which are passed to the native parquet reader.
            For example, you could use
            `nw.read_parquet('file.parquet', backend=pd, engine='pyarrow')`.

    Returns:
        DataFrame.
    """
    backend = cast("ModuleType | Implementation | str", backend)
    return _stableify(  # type: ignore[no-any-return]
        _read_parquet_impl(source, backend=backend, **kwargs)
    )


@deprecate_native_namespace(required=True)
def scan_parquet(
    source: str,
    *,
    backend: ModuleType | Implementation | str | None = None,
    native_namespace: ModuleType | None = None,  # noqa: ARG001
    **kwargs: Any,
) -> LazyFrame[Any]:
    """Lazily read from a parquet file.

    For the libraries that do not support lazy dataframes, the function reads
    a parquet file eagerly and then converts the resulting dataframe to a lazyframe.

    Note:
        Spark like backends require a session object to be passed in `kwargs`.

        For instance:

        ```py
        import narwhals as nw
        from sqlframe.duckdb import DuckDBSession

        nw.scan_parquet(source, backend="sqlframe", session=DuckDBSession())
        ```

    Arguments:
        source: Path to a file.
        backend: The eager backend for DataFrame creation.
            `backend` can be specified in various ways

            - As `Implementation.<BACKEND>` with `BACKEND` being `PANDAS`, `PYARROW`,
                `POLARS`, `MODIN`, `CUDF`, `PYSPARK` or `SQLFRAME`.
            - As a string: `"pandas"`, `"pyarrow"`, `"polars"`, `"modin"`, `"cudf"`,
                `"pyspark"` or `"sqlframe"`.
            - Directly as a module `pandas`, `pyarrow`, `polars`, `modin`, `cudf`,
                `pyspark.sql` or `sqlframe`.
        native_namespace: The native library to use for DataFrame creation.

            *Deprecated* (v1.31.0)

            Please use `backend` instead. Note that `native_namespace` is still available
            (and won't emit a deprecation warning) if you use `narwhals.stable.v1`,
            see [perfect backwards compatibility policy](../backcompat.md/).
        kwargs: Extra keyword arguments which are passed to the native parquet reader.
            For example, you could use
            `nw.scan_parquet('file.parquet', backend=pd, engine='pyarrow')`.

    Returns:
        LazyFrame.
    """
    backend = cast("ModuleType | Implementation | str", backend)
    return _stableify(  # type: ignore[no-any-return]
        _scan_parquet_impl(source, backend=backend, **kwargs)
    )


__all__ = [
    "Array",
    "Binary",
    "Boolean",
    "Categorical",
    "DataFrame",
    "Date",
    "Datetime",
    "Decimal",
    "Duration",
    "Enum",
    "Expr",
    "Field",
    "Float32",
    "Float64",
    "Implementation",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "Int128",
    "LazyFrame",
    "List",
    "Object",
    "Schema",
    "Series",
    "String",
    "Struct",
    "Time",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "UInt128",
    "Unknown",
    "all",
    "all_horizontal",
    "any_horizontal",
    "col",
    "concat",
    "concat_str",
    "dependencies",
    "dtypes",
    "exceptions",
    "exclude",
    "from_arrow",
    "from_dict",
    "from_native",
    "from_numpy",
    "generate_temporary_column_name",
    "get_level",
    "get_native_namespace",
    "is_ordered_categorical",
    "len",
    "lit",
    "max",
    "max_horizontal",
    "maybe_align_index",
    "maybe_convert_dtypes",
    "maybe_get_index",
    "maybe_reset_index",
    "maybe_set_index",
    "mean",
    "mean_horizontal",
    "median",
    "min",
    "min_horizontal",
    "narwhalify",
    "new_series",
    "nth",
    "read_csv",
    "read_parquet",
    "scan_csv",
    "scan_parquet",
    "selectors",
    "show_versions",
    "sum",
    "sum_horizontal",
    "to_native",
    "to_py_scalar",
    "when",
]
