from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from functools import wraps
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import TypeVar
from typing import overload

from narwhals.dependencies import get_cudf
from narwhals.dependencies import get_cupy
from narwhals.dependencies import get_dask
from narwhals.dependencies import get_dask_expr
from narwhals.dependencies import get_modin
from narwhals.dependencies import get_numpy
from narwhals.dependencies import get_pandas
from narwhals.dependencies import get_polars
from narwhals.dependencies import get_pyarrow
from narwhals.dependencies import is_cudf_dataframe
from narwhals.dependencies import is_cudf_series
from narwhals.dependencies import is_dask_dataframe
from narwhals.dependencies import is_duckdb_relation
from narwhals.dependencies import is_ibis_table
from narwhals.dependencies import is_modin_dataframe
from narwhals.dependencies import is_modin_series
from narwhals.dependencies import is_pandas_dataframe
from narwhals.dependencies import is_pandas_series
from narwhals.dependencies import is_polars_dataframe
from narwhals.dependencies import is_polars_lazyframe
from narwhals.dependencies import is_polars_series
from narwhals.dependencies import is_pyarrow_chunked_array
from narwhals.dependencies import is_pyarrow_table

if TYPE_CHECKING:
    from narwhals.dataframe import DataFrame
    from narwhals.dataframe import LazyFrame
    from narwhals.series import Series
    from narwhals.typing import DTypes
    from narwhals.typing import IntoDataFrameT
    from narwhals.typing import IntoFrameT
    from narwhals.typing import IntoSeriesT

T = TypeVar("T")

NON_TEMPORAL_SCALAR_TYPES = (
    bool,
    bytes,
    str,
    int,
    float,
    complex,
)


@overload
def to_native(
    narwhals_object: DataFrame[IntoDataFrameT], *, pass_through: Literal[False] = ...
) -> IntoDataFrameT: ...
@overload
def to_native(
    narwhals_object: LazyFrame[IntoFrameT], *, pass_through: Literal[False] = ...
) -> IntoFrameT: ...
@overload
def to_native(narwhals_object: Series, *, pass_through: Literal[False] = ...) -> Any: ...
@overload
def to_native(narwhals_object: Any, *, pass_through: bool) -> Any: ...


def to_native(
    narwhals_object: DataFrame[IntoFrameT] | LazyFrame[IntoFrameT] | Series,
    *,
    strict: bool | None = None,
    pass_through: bool | None = None,
) -> IntoFrameT | Any:
    """
    Convert Narwhals object to native one.

    Arguments:
        narwhals_object: Narwhals object.
        strict: whether to raise on non-Narwhals input.

    Returns:
        Object of class that user started with.
    """
    from narwhals.dataframe import BaseFrame
    from narwhals.series import Series
    from narwhals.utils import validate_strict_and_pass_though

    pass_through = validate_strict_and_pass_though(
        strict, pass_through, pass_through_default=False, emit_deprecation_warning=True
    )

    if isinstance(narwhals_object, BaseFrame):
        return narwhals_object._compliant_frame._native_frame
    if isinstance(narwhals_object, Series):
        return narwhals_object._compliant_series._native_series

    if not pass_through:
        msg = f"Expected Narwhals object, got {type(narwhals_object)}."
        raise TypeError(msg)
    return narwhals_object


@overload
def from_native(
    native_object: IntoDataFrameT | IntoSeriesT,
    *,
    pass_through: Literal[True],
    eager_only: None = ...,
    eager_or_interchange_only: Literal[True],
    series_only: None = ...,
    allow_series: Literal[True],
) -> DataFrame[IntoDataFrameT]: ...


@overload
def from_native(
    native_object: IntoDataFrameT | IntoSeriesT,
    *,
    pass_through: Literal[True],
    eager_only: Literal[True],
    eager_or_interchange_only: None = ...,
    series_only: None = ...,
    allow_series: Literal[True],
) -> DataFrame[IntoDataFrameT] | Series: ...


@overload
def from_native(
    native_object: IntoDataFrameT,
    *,
    pass_through: Literal[True],
    eager_only: None = ...,
    eager_or_interchange_only: Literal[True],
    series_only: None = ...,
    allow_series: None = ...,
) -> DataFrame[IntoDataFrameT]: ...


@overload
def from_native(
    native_object: T,
    *,
    pass_through: Literal[True],
    eager_only: None = ...,
    eager_or_interchange_only: Literal[True],
    series_only: None = ...,
    allow_series: None = ...,
) -> T: ...


@overload
def from_native(
    native_object: IntoDataFrameT,
    *,
    pass_through: Literal[True],
    eager_only: Literal[True],
    eager_or_interchange_only: None = ...,
    series_only: None = ...,
    allow_series: None = ...,
) -> DataFrame[IntoDataFrameT]: ...


@overload
def from_native(
    native_object: T,
    *,
    pass_through: Literal[True],
    eager_only: Literal[True],
    eager_or_interchange_only: None = ...,
    series_only: None = ...,
    allow_series: None = ...,
) -> T: ...


@overload
def from_native(
    native_object: IntoFrameT | IntoSeriesT,
    *,
    pass_through: Literal[True],
    eager_only: None = ...,
    eager_or_interchange_only: None = ...,
    series_only: None = ...,
    allow_series: Literal[True],
) -> DataFrame[IntoFrameT] | LazyFrame[IntoFrameT] | Series: ...


@overload
def from_native(
    native_object: IntoSeriesT,
    *,
    pass_through: Literal[True],
    eager_only: None = ...,
    eager_or_interchange_only: None = ...,
    series_only: Literal[True],
    allow_series: None = ...,
) -> Series: ...


@overload
def from_native(
    native_object: IntoFrameT,
    *,
    pass_through: Literal[True],
    eager_only: None = ...,
    eager_or_interchange_only: None = ...,
    series_only: None = ...,
    allow_series: None = ...,
) -> DataFrame[IntoFrameT] | LazyFrame[IntoFrameT]: ...


@overload
def from_native(
    native_object: T,
    *,
    pass_through: Literal[True],
    eager_only: None = ...,
    eager_or_interchange_only: None = ...,
    series_only: None = ...,
    allow_series: None = ...,
) -> T: ...


@overload
def from_native(
    native_object: IntoDataFrameT,
    *,
    pass_through: Literal[False] = ...,
    eager_only: None = ...,
    eager_or_interchange_only: Literal[True],
    series_only: None = ...,
    allow_series: None = ...,
) -> DataFrame[IntoDataFrameT]:
    """
    from_native(df, pass_through=False, eager_or_interchange_only=True)
    from_native(df, eager_or_interchange_only=True)
    """


@overload
def from_native(
    native_object: IntoDataFrameT,
    *,
    pass_through: Literal[False] = ...,
    eager_only: Literal[True],
    eager_or_interchange_only: None = ...,
    series_only: None = ...,
    allow_series: None = ...,
) -> DataFrame[IntoDataFrameT]:
    """
    from_native(df, pass_through=False, eager_only=True)
    from_native(df, eager_only=True)
    """


@overload
def from_native(
    native_object: IntoFrameT | IntoSeriesT,
    *,
    pass_through: Literal[False] = ...,
    eager_only: None = ...,
    eager_or_interchange_only: None = ...,
    series_only: None = ...,
    allow_series: Literal[True],
) -> DataFrame[Any] | LazyFrame[Any] | Series:
    """
    from_native(df, pass_through=False, allow_series=True)
    from_native(df, allow_series=True)
    """


@overload
def from_native(
    native_object: IntoSeriesT,
    *,
    pass_through: Literal[False] = ...,
    eager_only: None = ...,
    eager_or_interchange_only: None = ...,
    series_only: Literal[True],
    allow_series: None = ...,
) -> Series:
    """
    from_native(df, pass_through=False, series_only=True)
    from_native(df, series_only=True)
    """


@overload
def from_native(
    native_object: IntoFrameT,
    *,
    pass_through: Literal[False] = ...,
    eager_only: None = ...,
    eager_or_interchange_only: None = ...,
    series_only: None = ...,
    allow_series: None = ...,
) -> DataFrame[IntoFrameT] | LazyFrame[IntoFrameT]:
    """
    from_native(df, pass_through=False)
    from_native(df)
    """


# All params passed in as variables
@overload
def from_native(
    native_object: Any,
    *,
    pass_through: bool,
    eager_only: bool | None,
    eager_or_interchange_only: bool | None = None,
    series_only: bool | None,
    allow_series: bool | None,
) -> Any: ...


def from_native(
    native_object: Any,
    *,
    strict: bool | None = None,
    pass_through: bool | None = None,
    eager_only: bool | None = None,
    eager_or_interchange_only: bool | None = None,
    series_only: bool | None = None,
    allow_series: bool | None = None,
) -> Any:
    """
    Convert dataframe/series to Narwhals DataFrame, LazyFrame, or Series.

    Arguments:
        native_object: Raw object from user.
            Depending on the other arguments, input object can be:

            - pandas.DataFrame
            - polars.DataFrame
            - polars.LazyFrame
            - anything with a `__narwhals_dataframe__` or `__narwhals_lazyframe__` method
            - pandas.Series
            - polars.Series
            - anything with a `__narwhals_series__` method
        strict: Determine what happens if the object isn't supported by Narwhals:

            - `True` (default): raise an error
            - `False`: pass object through as-is

            **Deprecated** (v1.13.0):
                Please use `pass_through` instead. Note that `strict` is still available
                (and won't emit a deprecation warning) if you use `narwhals.stable.v1`,
                see [perfect backwards compatibility policy](https://narwhals-dev.github.io/narwhals/backcompat/).
        pass_through: Determine what happens if the object isn't supported by Narwhals:

            - `False` (default): raise an error
            - `True`: pass object through as-is
        eager_only: Whether to only allow eager objects.
        eager_or_interchange_only: Whether to only allow eager objects or objects which
            implement the Dataframe Interchange Protocol.
        series_only: Whether to only allow series.
        allow_series: Whether to allow series (default is only dataframe / lazyframe).

    Returns:
        narwhals.DataFrame or narwhals.LazyFrame or narwhals.Series
    """
    from narwhals import dtypes
    from narwhals.utils import validate_strict_and_pass_though

    pass_through = validate_strict_and_pass_though(
        strict, pass_through, pass_through_default=False, emit_deprecation_warning=True
    )

    return _from_native_impl(
        native_object,
        pass_through=pass_through,
        eager_only=eager_only,
        eager_or_interchange_only=eager_or_interchange_only,
        series_only=series_only,
        allow_series=allow_series,
        dtypes=dtypes,  # type: ignore[arg-type]
    )


def _from_native_impl(  # noqa: PLR0915
    native_object: Any,
    *,
    pass_through: bool = False,
    eager_only: bool | None = None,
    eager_or_interchange_only: bool | None = None,
    series_only: bool | None = None,
    allow_series: bool | None = None,
    dtypes: DTypes,
) -> Any:
    from narwhals._arrow.dataframe import ArrowDataFrame
    from narwhals._arrow.series import ArrowSeries
    from narwhals._dask.dataframe import DaskLazyFrame
    from narwhals._duckdb.dataframe import DuckDBInterchangeFrame
    from narwhals._ibis.dataframe import IbisInterchangeFrame
    from narwhals._interchange.dataframe import InterchangeFrame
    from narwhals._pandas_like.dataframe import PandasLikeDataFrame
    from narwhals._pandas_like.series import PandasLikeSeries
    from narwhals._polars.dataframe import PolarsDataFrame
    from narwhals._polars.dataframe import PolarsLazyFrame
    from narwhals._polars.series import PolarsSeries
    from narwhals.dataframe import DataFrame
    from narwhals.dataframe import LazyFrame
    from narwhals.series import Series
    from narwhals.utils import Implementation
    from narwhals.utils import parse_version

    # Early returns
    if isinstance(native_object, (DataFrame, LazyFrame)) and not series_only:
        return native_object
    if isinstance(native_object, Series) and (series_only or allow_series):
        return native_object

    if series_only:
        if allow_series is False:
            msg = "Invalid parameter combination: `series_only=True` and `allow_series=False`"
            raise ValueError(msg)
        allow_series = True
    if eager_only and eager_or_interchange_only:
        msg = "Invalid parameter combination: `eager_only=True` and `eager_or_interchange_only=True`"
        raise ValueError(msg)

    # Extensions
    if hasattr(native_object, "__narwhals_dataframe__"):
        if series_only:
            if not pass_through:
                msg = "Cannot only use `series_only` with dataframe"
                raise TypeError(msg)
            return native_object
        return DataFrame(
            native_object.__narwhals_dataframe__(),
            level="full",
        )
    elif hasattr(native_object, "__narwhals_lazyframe__"):
        if series_only:
            if not pass_through:
                msg = "Cannot only use `series_only` with lazyframe"
                raise TypeError(msg)
            return native_object
        if eager_only or eager_or_interchange_only:
            if not pass_through:
                msg = "Cannot only use `eager_only` or `eager_or_interchange_only` with lazyframe"
                raise TypeError(msg)
            return native_object
        return LazyFrame(
            native_object.__narwhals_lazyframe__(),
            level="full",
        )
    elif hasattr(native_object, "__narwhals_series__"):
        if not allow_series:
            if not pass_through:
                msg = "Please set `allow_series=True` or `series_only=True`"
                raise TypeError(msg)
            return native_object
        return Series(
            native_object.__narwhals_series__(),
            level="full",
        )

    # Polars
    elif is_polars_dataframe(native_object):
        if series_only:
            if not pass_through:
                msg = "Cannot only use `series_only` with polars.DataFrame"
                raise TypeError(msg)
            return native_object
        pl = get_polars()
        return DataFrame(
            PolarsDataFrame(
                native_object,
                backend_version=parse_version(pl.__version__),
                dtypes=dtypes,
            ),
            level="full",
        )
    elif is_polars_lazyframe(native_object):
        if series_only:
            if not pass_through:
                msg = "Cannot only use `series_only` with polars.LazyFrame"
                raise TypeError(msg)
            return native_object
        if eager_only or eager_or_interchange_only:
            if not pass_through:
                msg = "Cannot only use `eager_only` or `eager_or_interchange_only` with polars.LazyFrame"
                raise TypeError(msg)
            return native_object
        pl = get_polars()
        return LazyFrame(
            PolarsLazyFrame(
                native_object,
                backend_version=parse_version(pl.__version__),
                dtypes=dtypes,
            ),
            level="full",
        )
    elif is_polars_series(native_object):
        pl = get_polars()
        if not allow_series:
            if not pass_through:
                msg = "Please set `allow_series=True` or `series_only=True`"
                raise TypeError(msg)
            return native_object
        return Series(
            PolarsSeries(
                native_object,
                backend_version=parse_version(pl.__version__),
                dtypes=dtypes,
            ),
            level="full",
        )

    # pandas
    elif is_pandas_dataframe(native_object):
        if series_only:
            if not pass_through:
                msg = "Cannot only use `series_only` with dataframe"
                raise TypeError(msg)
            return native_object
        pd = get_pandas()
        return DataFrame(
            PandasLikeDataFrame(
                native_object,
                backend_version=parse_version(pd.__version__),
                implementation=Implementation.PANDAS,
                dtypes=dtypes,
            ),
            level="full",
        )
    elif is_pandas_series(native_object):
        if not allow_series:
            if not pass_through:
                msg = "Please set `allow_series=True` or `series_only=True`"
                raise TypeError(msg)
            return native_object
        pd = get_pandas()
        return Series(
            PandasLikeSeries(
                native_object,
                implementation=Implementation.PANDAS,
                backend_version=parse_version(pd.__version__),
                dtypes=dtypes,
            ),
            level="full",
        )

    # Modin
    elif is_modin_dataframe(native_object):  # pragma: no cover
        mpd = get_modin()
        if series_only:
            if not pass_through:
                msg = "Cannot only use `series_only` with modin.DataFrame"
                raise TypeError(msg)
            return native_object
        return DataFrame(
            PandasLikeDataFrame(
                native_object,
                implementation=Implementation.MODIN,
                backend_version=parse_version(mpd.__version__),
                dtypes=dtypes,
            ),
            level="full",
        )
    elif is_modin_series(native_object):  # pragma: no cover
        mpd = get_modin()
        if not allow_series:
            if not pass_through:
                msg = "Please set `allow_series=True` or `series_only=True`"
                raise TypeError(msg)
            return native_object
        return Series(
            PandasLikeSeries(
                native_object,
                implementation=Implementation.MODIN,
                backend_version=parse_version(mpd.__version__),
                dtypes=dtypes,
            ),
            level="full",
        )

    # cuDF
    elif is_cudf_dataframe(native_object):  # pragma: no cover
        cudf = get_cudf()
        if series_only:
            if not pass_through:
                msg = "Cannot only use `series_only` with cudf.DataFrame"
                raise TypeError(msg)
            return native_object
        return DataFrame(
            PandasLikeDataFrame(
                native_object,
                implementation=Implementation.CUDF,
                backend_version=parse_version(cudf.__version__),
                dtypes=dtypes,
            ),
            level="full",
        )
    elif is_cudf_series(native_object):  # pragma: no cover
        cudf = get_cudf()
        if not allow_series:
            if not pass_through:
                msg = "Please set `allow_series=True` or `series_only=True`"
                raise TypeError(msg)
            return native_object
        return Series(
            PandasLikeSeries(
                native_object,
                implementation=Implementation.CUDF,
                backend_version=parse_version(cudf.__version__),
                dtypes=dtypes,
            ),
            level="full",
        )

    # PyArrow
    elif is_pyarrow_table(native_object):
        pa = get_pyarrow()
        if series_only:
            if not pass_through:
                msg = "Cannot only use `series_only` with arrow table"
                raise TypeError(msg)
            return native_object
        return DataFrame(
            ArrowDataFrame(
                native_object,
                backend_version=parse_version(pa.__version__),
                dtypes=dtypes,
            ),
            level="full",
        )
    elif is_pyarrow_chunked_array(native_object):
        pa = get_pyarrow()
        if not allow_series:
            if not pass_through:
                msg = "Please set `allow_series=True` or `series_only=True`"
                raise TypeError(msg)
            return native_object
        return Series(
            ArrowSeries(
                native_object,
                backend_version=parse_version(pa.__version__),
                name="",
                dtypes=dtypes,
            ),
            level="full",
        )

    # Dask
    elif is_dask_dataframe(native_object):
        if series_only:
            if not pass_through:
                msg = "Cannot only use `series_only` with dask DataFrame"
                raise TypeError(msg)
            return native_object
        if eager_only or eager_or_interchange_only:
            if not pass_through:
                msg = "Cannot only use `eager_only` or `eager_or_interchange_only` with dask DataFrame"
                raise TypeError(msg)
            return native_object
        if get_dask_expr() is None:  # pragma: no cover
            msg = "Please install dask-expr"
            raise ImportError(msg)
        return LazyFrame(
            DaskLazyFrame(
                native_object,
                backend_version=parse_version(get_dask().__version__),
                dtypes=dtypes,
            ),
            level="full",
        )

    # DuckDB
    elif is_duckdb_relation(native_object):
        if eager_only or series_only:  # pragma: no cover
            if not pass_through:
                msg = (
                    "Cannot only use `series_only=True` or `eager_only=False` "
                    "with DuckDB Relation"
                )
            else:
                return native_object
            raise TypeError(msg)
        return DataFrame(
            DuckDBInterchangeFrame(native_object, dtypes=dtypes),
            level="interchange",
        )

    # Ibis
    elif is_ibis_table(native_object):  # pragma: no cover
        if eager_only or series_only:
            if not pass_through:
                msg = (
                    "Cannot only use `series_only=True` or `eager_only=False` "
                    "with Ibis table"
                )
                raise TypeError(msg)
            return native_object
        return DataFrame(
            IbisInterchangeFrame(native_object, dtypes=dtypes),
            level="interchange",
        )

    # Interchange protocol
    elif hasattr(native_object, "__dataframe__"):
        if eager_only or series_only:
            if not pass_through:
                msg = (
                    "Cannot only use `series_only=True` or `eager_only=False` "
                    "with object which only implements __dataframe__"
                )
                raise TypeError(msg)
            return native_object
        return DataFrame(
            InterchangeFrame(native_object, dtypes=dtypes),
            level="interchange",
        )

    elif not pass_through:
        msg = f"Expected pandas-like dataframe, Polars dataframe, or Polars lazyframe, got: {type(native_object)}"
        raise TypeError(msg)
    return native_object


def get_native_namespace(obj: Any) -> Any:
    """
    Get native namespace from object.

    Examples:
        >>> import polars as pl
        >>> import pandas as pd
        >>> import narwhals as nw
        >>> df = nw.from_native(pd.DataFrame({"a": [1, 2, 3]}))
        >>> nw.get_native_namespace(df)
        <module 'pandas'...>
        >>> df = nw.from_native(pl.DataFrame({"a": [1, 2, 3]}))
        >>> nw.get_native_namespace(df)
        <module 'polars'...>
    """
    return obj.__native_namespace__()


def narwhalify(
    func: Callable[..., Any] | None = None,
    *,
    strict: bool | None = None,
    pass_through: bool | None = None,
    eager_only: bool | None = False,
    eager_or_interchange_only: bool | None = False,
    series_only: bool | None = False,
    allow_series: bool | None = True,
) -> Callable[..., Any]:
    """
    Decorate function so it becomes dataframe-agnostic.

    `narwhalify` will try to convert any dataframe/series-like object into the narwhal
    respective DataFrame/Series, while leaving the other parameters as they are.

    Similarly, if the output of the function is a narwhals DataFrame or Series, it will be
    converted back to the original dataframe/series type, while if the output is another
    type it will be left as is.

    By setting `strict=True`, then every input and every output will be required to be a
    dataframe/series-like object.

    Instead of writing

    ```python
    import narwhals as nw


    def func(df):
        df = nw.from_native(df, strict=False)
        df = df.group_by("a").agg(nw.col("b").sum())
        return nw.to_native(df)
    ```

    you can just write

    ```python
    import narwhals as nw


    @nw.narwhalify
    def func(df):
        return df.group_by("a").agg(nw.col("b").sum())
    ```

    You can also pass in extra arguments, e.g.

    ```python
    @nw.narwhalify(eager_only=True)
    ```

    that will get passed down to `nw.from_native`.

    Arguments:
        func: Function to wrap in a `from_native`-`to_native` block.
        strict: Whether to raise if object can't be converted or to just leave it as-is
            (default).
        eager_only: Whether to only allow eager objects.
        eager_or_interchange_only: Whether to only allow eager objects or objects which
            implement the Dataframe Interchange Protocol.
        series_only: Whether to only allow series.
        allow_series: Whether to allow series (default is only dataframe / lazyframe).
    """
    from narwhals.utils import validate_strict_and_pass_though

    pass_through = validate_strict_and_pass_though(
        strict, pass_through, pass_through_default=True, emit_deprecation_warning=True
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

            if len(backends) > 1:
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


def to_py_scalar(scalar_like: Any) -> Any:
    """If a scalar is not Python native, converts it to Python native.

    Raises:
        ValueError: If the object is not convertible to a scalar.

    Examples:
        >>> import narwhals as nw
        >>> import pandas as pd
        >>> df = nw.from_native(pd.DataFrame({"a": [1, 2, 3]}))
        >>> nw.to_py_scalar(df["a"].item(0))
        1
        >>> import pyarrow as pa
        >>> df = nw.from_native(pa.table({"a": [1, 2, 3]}))
        >>> nw.to_py_scalar(df["a"].item(0))
        1
        >>> nw.to_py_scalar(1)
        1
    """
    if scalar_like is None:
        return None
    if isinstance(scalar_like, NON_TEMPORAL_SCALAR_TYPES):
        return scalar_like

    np = get_numpy()
    if (
        np
        and isinstance(scalar_like, np.datetime64)
        and scalar_like.dtype == "datetime64[ns]"
    ):
        return datetime(1970, 1, 1) + timedelta(microseconds=scalar_like.item() // 1000)

    if np and np.isscalar(scalar_like) and hasattr(scalar_like, "item"):
        return scalar_like.item()

    pd = get_pandas()
    if pd and isinstance(scalar_like, pd.Timestamp):
        return scalar_like.to_pydatetime()
    if pd and isinstance(scalar_like, pd.Timedelta):
        return scalar_like.to_pytimedelta()
    if pd and pd.api.types.is_scalar(scalar_like):
        try:
            is_na = pd.isna(scalar_like)
        except Exception:  # pragma: no cover  # noqa: BLE001, S110
            pass
        else:
            if is_na:
                return None

    # pd.Timestamp and pd.Timedelta subclass datetime and timedelta,
    # so we need to check this separately
    if isinstance(scalar_like, (datetime, timedelta)):
        return scalar_like

    pa = get_pyarrow()
    if pa and isinstance(scalar_like, pa.Scalar):
        return scalar_like.as_py()

    cupy = get_cupy()
    if (  # pragma: no cover
        cupy and isinstance(scalar_like, cupy.ndarray) and scalar_like.size == 1
    ):
        return scalar_like.item()

    msg = (
        f"Expected object convertible to a scalar, found {type(scalar_like)}. "
        "Please report a bug to https://github.com/narwhals-dev/narwhals/issues"
    )
    raise ValueError(msg)


__all__ = [
    "get_native_namespace",
    "to_native",
    "narwhalify",
    "to_py_scalar",
]
