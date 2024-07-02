from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import TypeVar

from narwhals.dependencies import get_cudf
from narwhals.dependencies import get_modin
from narwhals.dependencies import get_numpy
from narwhals.dependencies import get_pandas
from narwhals.utils import flatten
from narwhals.utils import isinstance_or_issubclass
from narwhals.utils import parse_version

T = TypeVar("T")

if TYPE_CHECKING:
    from narwhals._pandas_like.dataframe import PandasDataFrame
    from narwhals._pandas_like.expr import PandasExpr
    from narwhals._pandas_like.series import PandasSeries
    from narwhals.dtypes import DType

    ExprT = TypeVar("ExprT", bound=PandasExpr)

    from narwhals._arrow.typing import IntoArrowExpr
    from narwhals._pandas_like.typing import IntoPandasExpr


def validate_column_comparand(index: Any, other: Any) -> Any:
    """Validate RHS of binary operation.

    If the comparison isn't supported, return `NotImplemented` so that the
    "right-hand-side" operation (e.g. `__radd__`) can be tried.

    If RHS is length 1, return the scalar value, so that the underlying
    library can broadcast it.
    """
    from narwhals._pandas_like.dataframe import PandasDataFrame
    from narwhals._pandas_like.series import PandasSeries

    if isinstance(other, list):
        if len(other) > 1:
            # e.g. `plx.all() + plx.all()`
            msg = "Multi-output expressions are not supported in this context"
            raise ValueError(msg)
        other = other[0]
    if isinstance(other, PandasDataFrame):
        return NotImplemented
    if isinstance(other, PandasSeries):
        if other.len() == 1:
            # broadcast
            return other.item()
        if other._series.index is not index:
            return set_axis(other._series, index, implementation=other._implementation)
        return other._series
    return other


def validate_dataframe_comparand(index: Any, other: Any) -> Any:
    """Validate RHS of binary operation.

    If the comparison isn't supported, return `NotImplemented` so that the
    "right-hand-side" operation (e.g. `__radd__`) can be tried.
    """
    from narwhals._pandas_like.dataframe import PandasDataFrame
    from narwhals._pandas_like.series import PandasSeries

    if isinstance(other, PandasDataFrame):
        return NotImplemented
    if isinstance(other, PandasSeries):
        if other.len() == 1:
            # broadcast
            return other._series.item()
        if other._series.index is not index and not (other._series.index == index).all():
            return other._series.set_axis(index, axis=0)
        return other._series
    error_message = "Please report a bug"
    raise AssertionError(error_message)


def maybe_evaluate_expr(df: PandasDataFrame, expr: Any) -> Any:
    """Evaluate `expr` if it's an expression, otherwise return it as is."""
    from narwhals._pandas_like.expr import PandasExpr

    if isinstance(expr, PandasExpr):
        return expr._call(df)
    return expr


def parse_into_exprs(
    implementation: str,
    *exprs: IntoPandasExpr | Iterable[IntoPandasExpr],
    **named_exprs: IntoPandasExpr,
) -> list[PandasExpr]:
    """Parse each input as an expression (if it's not already one). See `parse_into_expr` for
    more details."""
    out = [parse_into_expr(implementation, into_expr) for into_expr in flatten(exprs)]
    for name, expr in named_exprs.items():
        out.append(parse_into_expr(implementation, expr).alias(name))
    return out


def parse_into_expr(
    implementation: str, into_expr: IntoPandasExpr | IntoArrowExpr
) -> PandasExpr:
    """Parse `into_expr` as an expression.

    For example, in Polars, we can do both `df.select('a')` and `df.select(pl.col('a'))`.
    We do the same in Narwhals:

    - if `into_expr` is already an expression, just return it
    - if it's a Series, then convert it to an expression
    - if it's a numpy array, then convert it to a Series and then to an expression
    - if it's a string, then convert it to an expression
    - else, raise
    """
    from narwhals._arrow.expr import ArrowExpr
    from narwhals._arrow.namespace import ArrowNamespace
    from narwhals._arrow.series import ArrowSeries
    from narwhals._pandas_like.expr import PandasExpr
    from narwhals._pandas_like.namespace import PandasNamespace
    from narwhals._pandas_like.series import PandasSeries

    if implementation == "arrow":
        plx: ArrowNamespace | PandasNamespace = ArrowNamespace()
    else:
        plx = PandasNamespace(implementation=implementation)
    if isinstance(into_expr, (PandasExpr, ArrowExpr)):
        return into_expr  # type: ignore[return-value]
    if isinstance(into_expr, (PandasSeries, ArrowSeries)):
        return plx._create_expr_from_series(into_expr)  # type: ignore[arg-type, return-value]
    if isinstance(into_expr, str):
        return plx.col(into_expr)  # type: ignore[return-value]
    if (np := get_numpy()) is not None and isinstance(into_expr, np.ndarray):
        series = create_native_series(into_expr, implementation=implementation)
        return plx._create_expr_from_series(series)  # type: ignore[arg-type, return-value]
    msg = f"Expected IntoExpr, got {type(into_expr)}"  # pragma: no cover
    raise AssertionError(msg)


def create_native_series(
    iterable: Any,
    implementation: str,
    index: Any = None,
) -> PandasSeries:
    from narwhals._pandas_like.series import PandasSeries

    if implementation == "pandas":
        pd = get_pandas()
        series = pd.Series(iterable, index=index, name="")
    elif implementation == "modin":
        mpd = get_modin()
        series = mpd.Series(iterable, index=index, name="")
    elif implementation == "cudf":
        cudf = get_cudf()
        series = cudf.Series(iterable, index=index, name="")
    return PandasSeries(series, implementation=implementation)


def evaluate_into_expr(
    df: PandasDataFrame, into_expr: IntoPandasExpr
) -> list[PandasSeries]:
    """
    Return list of raw columns.
    """
    expr = parse_into_expr(df._implementation, into_expr)
    return expr._call(df)


def evaluate_into_exprs(
    df: PandasDataFrame,
    *exprs: IntoPandasExpr,
    **named_exprs: IntoPandasExpr,
) -> list[PandasSeries]:
    """Evaluate each expr into Series."""
    series: list[PandasSeries] = [
        item
        for sublist in [evaluate_into_expr(df, into_expr) for into_expr in flatten(exprs)]
        for item in sublist
    ]
    for name, expr in named_exprs.items():
        evaluated_expr = evaluate_into_expr(df, expr)
        if len(evaluated_expr) > 1:
            msg = "Named expressions must return a single column"  # pragma: no cover
            raise AssertionError(msg)
        series.append(evaluated_expr[0].alias(name))
    return series


def reuse_series_implementation(
    expr: ExprT, attr: str, *args: Any, returns_scalar: bool = False, **kwargs: Any
) -> ExprT:
    """Reuse Series implementation for expression.

    If Series.foo is already defined, and we'd like Expr.foo to be the same, we can
    leverage this method to do that for us.

    Arguments
        expr: expression object.
        attr: name of method.
        returns_scalar: whether the Series version returns a scalar. In this case,
            the expression version should return a 1-row Series.
        args, kwargs: arguments and keyword arguments to pass to function.
    """
    plx = expr.__narwhals_namespace__()

    def func(df: PandasDataFrame) -> list[PandasSeries]:
        out: list[PandasSeries] = []
        for column in expr._call(df):
            _out = getattr(column, attr)(
                *[maybe_evaluate_expr(df, arg) for arg in args],
                **{
                    arg_name: maybe_evaluate_expr(df, arg_value)
                    for arg_name, arg_value in kwargs.items()
                },
            )
            if returns_scalar:
                out.append(plx._create_series_from_scalar(_out, column))
            else:
                out.append(_out)
        if expr._output_names is not None:  # safety check
            assert [s.name for s in out] == expr._output_names
        return out

    # Try tracking root and output names by combining them from all
    # expressions appearing in args and kwargs. If any anonymous
    # expression appears (e.g. nw.all()), then give up on tracking root names
    # and just set it to None.
    root_names = copy(expr._root_names)
    output_names = expr._output_names
    for arg in list(args) + list(kwargs.values()):
        if root_names is not None and isinstance(arg, expr.__class__):
            if arg._root_names is not None:
                root_names.extend(arg._root_names)
            else:
                root_names = None
                output_names = None
                break
        elif root_names is None:
            output_names = None
            break

    assert (output_names is None and root_names is None) or (
        output_names is not None and root_names is not None
    )  # safety check

    return plx._create_expr_from_callable(  # type: ignore[return-value]
        func,
        depth=expr._depth + 1,
        function_name=f"{expr._function_name}->{attr}",
        root_names=root_names,
        output_names=output_names,
    )


def reuse_series_namespace_implementation(
    expr: ExprT, namespace: str, attr: str, *args: Any, **kwargs: Any
) -> PandasExpr:
    """Just like `reuse_series_implementation`, but for e.g. `Expr.dt.foo` instead
    of `Expr.foo`.
    """
    from narwhals._pandas_like.expr import PandasExpr

    return PandasExpr(
        lambda df: [
            getattr(getattr(series, namespace), attr)(*args, **kwargs)
            for series in expr._call(df)
        ],
        depth=expr._depth + 1,
        function_name=f"{expr._function_name}->{namespace}.{attr}",
        root_names=expr._root_names,
        output_names=expr._output_names,
        implementation=expr._implementation,
    )


def is_simple_aggregation(expr: PandasExpr) -> bool:
    """
    Check if expr is a very simple one, such as:

    - nw.col('a').mean()  # depth 1
    - nw.mean('a')  # depth 1
    - nw.len()  # depth 0

    as opposed to, say

    - nw.col('a').filter(nw.col('b')>nw.col('c')).max()

    because then, we can use a fastpath in pandas.
    """
    return expr._depth < 2


def horizontal_concat(dfs: list[Any], implementation: str) -> Any:
    """
    Concatenate (native) DataFrames horizontally.

    Should be in namespace.
    """
    if implementation == "pandas":
        pd = get_pandas()

        if parse_version(pd.__version__) < parse_version("3.0.0"):
            return pd.concat(dfs, axis=1, copy=False)
        return pd.concat(dfs, axis=1)  # pragma: no cover
    if implementation == "cudf":  # pragma: no cover
        cudf = get_cudf()

        return cudf.concat(dfs, axis=1)
    if implementation == "modin":  # pragma: no cover
        mpd = get_modin()

        return mpd.concat(dfs, axis=1)
    msg = f"Unknown implementation: {implementation}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


def vertical_concat(dfs: list[Any], implementation: str) -> Any:
    """
    Concatenate (native) DataFrames vertically.

    Should be in namespace.
    """
    if not dfs:
        msg = "No dataframes to concatenate"  # pragma: no cover
        raise AssertionError(msg)
    cols = set(dfs[0].columns)
    for df in dfs:
        cols_current = set(df.columns)
        if cols_current != cols:
            msg = "unable to vstack, column names don't match"
            raise TypeError(msg)
    if implementation == "pandas":
        pd = get_pandas()

        if parse_version(pd.__version__) < parse_version("3.0.0"):
            return pd.concat(dfs, axis=0, copy=False)
        return pd.concat(dfs, axis=0)  # pragma: no cover
    if implementation == "cudf":  # pragma: no cover
        cudf = get_cudf()

        return cudf.concat(dfs, axis=0)
    if implementation == "modin":  # pragma: no cover
        mpd = get_modin()

        return mpd.concat(dfs, axis=0)
    msg = f"Unknown implementation: {implementation}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


def native_series_from_iterable(
    data: Iterable[Any], name: str, index: Any, implementation: str
) -> Any:
    """Return native series."""
    if implementation == "pandas":
        pd = get_pandas()

        return pd.Series(data, name=name, index=index, copy=False)
    if implementation == "cudf":  # pragma: no cover
        cudf = get_cudf()

        return cudf.Series(data, name=name, index=index)
    if implementation == "modin":  # pragma: no cover
        mpd = get_modin()

        return mpd.Series(data, name=name, index=index)
    msg = f"Unknown implementation: {implementation}"  # pragma: no cover
    raise TypeError(msg)  # pragma: no cover


def set_axis(obj: T, index: Any, implementation: str) -> T:
    if implementation == "pandas" and parse_version(
        get_pandas().__version__
    ) >= parse_version("1.5.0"):
        return obj.set_axis(index, axis=0, copy=False)  # type: ignore[no-any-return, attr-defined]
    else:  # pragma: no cover
        return obj.set_axis(index, axis=0)  # type: ignore[no-any-return, attr-defined]


def __translate_primitive_dtype(dtype: str) -> DType | None:
    from narwhals import dtypes

    dtype_mappers: dict[str, type[dtypes.DType]] = {
        "int64": dtypes.Int64,
        "int32": dtypes.Int32,
        "int16": dtypes.Int16,
        "int8": dtypes.Int8,
        "uint64": dtypes.UInt64,
        "uint32": dtypes.UInt32,
        "uint16": dtypes.UInt16,
        "uint8": dtypes.UInt8,
        "float64": dtypes.Float64,
        "double": dtypes.Float64,
        "float32": dtypes.Float32,
        "float": dtypes.Float32,
        "string": dtypes.String,
        "string[python]": dtypes.String,
        "large_string": dtypes.String,
        "bool": dtypes.Boolean,
        "boolean": dtypes.Boolean,
        "category": dtypes.Categorical,
        "date32[day]": dtypes.Date,
    }

    if dtype not in dtype_mappers:
        return None

    dtype_factory = next(
        dtype_factory
        for pandas_dtype, dtype_factory in dtype_mappers.items()
        if dtype == pandas_dtype
    )
    return dtype_factory()


def __translate_datetime_dtype(dtype: str) -> DType | None:
    from narwhals import dtypes

    if dtype in ("category",) or dtype.startswith("dictionary<"):
        return dtypes.Categorical()
    if dtype.startswith("datetime64"):
        # todo: different time units and time zones
        return dtypes.Datetime()
    if dtype.startswith(("timedelta64", "duration")):
        # todo: different time units
        return dtypes.Duration()
    if dtype.startswith("timestamp["):
        # pyarrow-backed datetime
        # todo: different time units and time zones
        return dtypes.Datetime()

    return None


def __translate_object_dtype(dtype: str, column: Any) -> DType | None:
    from narwhals import dtypes

    if str(dtype) != "object":
        return None

    if (idx := column.first_valid_index()) is not None and isinstance(
        column.loc[idx], str
    ):
        # Infer based on first non-missing value.
        # For pandas pre 3.0, this isn't perfect.
        # After pandas 3.0, pandas has a dedicated string dtype
        # which is inferred by default.
        return dtypes.String()
    return dtypes.Object()


def translate_dtype(column: Any) -> DType:
    dtype = str(column.dtype).lower().replace("[pyarrow]", "")

    primitive_dtype = __translate_primitive_dtype(dtype)
    if primitive_dtype:
        return primitive_dtype

    datetime_dtype = __translate_datetime_dtype(dtype)
    if datetime_dtype:
        return datetime_dtype

    object_dtype = __translate_object_dtype(dtype, column)
    if object_dtype:
        return object_dtype

    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def get_dtype_backend(dtype: Any, implementation: str) -> str:
    if implementation == "pandas":
        pd = get_pandas()
        if hasattr(pd, "ArrowDtype") and isinstance(dtype, pd.ArrowDtype):
            return "pyarrow-nullable"

        try:
            if isinstance(dtype, pd.core.dtypes.dtypes.BaseMaskedDtype):
                return "pandas-nullable"
        except AttributeError:  # pragma: no cover
            # defensive check for old pandas versions
            pass
        return "numpy"
    else:  # pragma: no cover
        return "numpy"


def reverse_translate_dtype(
    dtype: DType | type[DType], starting_dtype: Any, implementation: str
) -> Any:
    from narwhals import dtypes

    dtypes_mapping: dict[type[dtypes.DType], dict[str, str | Any]] = {
        dtypes.Float64: {
            "pyarrow-nullable": "Float64[pyarrow]",
            "pandas-nullable": "Float64",
            "numpy": "float64",
        },
        dtypes.Float32: {
            "pyarrow-nullable": "Float32[pyarrow]",
            "pandas-nullable": "Float32",
            "numpy": "float32",
        },
        dtypes.Int64: {
            "pyarrow-nullable": "Int64[pyarrow]",
            "pandas-nullable": "Int64",
            "numpy": "int64",
        },
        dtypes.Int32: {
            "pyarrow-nullable": "Int32[pyarrow]",
            "pandas-nullable": "Int32",
            "numpy": "int32",
        },
        dtypes.Int16: {
            "pyarrow-nullable": "Int16[pyarrow]",
            "pandas-nullable": "Int16",
            "numpy": "int16",
        },
        dtypes.Int8: {
            "pyarrow-nullable": "Int8[pyarrow]",
            "pandas-nullable": "Int8",
            "numpy": "int8",
        },
        dtypes.UInt64: {
            "pyarrow-nullable": "UInt64[pyarrow]",
            "pandas-nullable": "UInt64",
            "numpy": "uint64",
        },
        dtypes.UInt32: {
            "pyarrow-nullable": "UInt32[pyarrow]",
            "pandas-nullable": "UInt32",
            "numpy": "uint32",
        },
        dtypes.UInt16: {
            "pyarrow-nullable": "UInt16[pyarrow]",
            "pandas-nullable": "UInt16",
            "numpy": "uint16",
        },
        dtypes.UInt8: {
            "pyarrow-nullable": "UInt8[pyarrow]",
            "pandas-nullable": "UInt8",
            "numpy": "uint8",
        },
        dtypes.String: {
            "pyarrow-nullable": "string[pyarrow]",
            "pandas-nullable": "string",
            "numpy": str,
        },
        dtypes.Boolean: {
            "pyarrow-nullable": "boolean[pyarrow]",
            "pandas-nullable": "boolean",
            "numpy": "bool",
        },
        dtypes.Categorical: {
            "pyarrow-nullable": "category",
            "pandas-nullable": "category",
            "numpy": "category",
        },
        dtypes.Datetime: {
            "pyarrow-nullable": "timestamp[ns][pyarrow]",
            "pandas-nullable": "datetime64[ns]",
            "numpy": "datetime64[ns]",
        },
        dtypes.Duration: {
            "pyarrow-nullable": "duration[ns][pyarrow]",
            "pandas-nullable": "timedelta64[ns]",
            "numpy": "timedelta64[ns]",
        },
        dtypes.Date: {
            "pyarrow-nullable": "date32[pyarrow]",
        },
    }

    dtype_backend = get_dtype_backend(starting_dtype, implementation)

    for dtype_type, dtype_to_backend_mapping in dtypes_mapping.items():
        if not isinstance_or_issubclass(dtype, dtype_type):
            continue

        if dtype_backend not in dtype_to_backend_mapping:
            msg = f"Type {dtype_type} not supported for backend {dtype_backend}"
            raise NotImplementedError(msg)

        return dtype_to_backend_mapping[dtype_backend]

    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def validate_indices(series: list[PandasSeries]) -> list[Any]:
    idx = series[0]._series.index
    reindexed = [series[0]._series]
    for s in series[1:]:
        if s._series.index is not idx:
            reindexed.append(s._series.set_axis(idx.rename(s._series.index.name), axis=0))
        else:
            reindexed.append(s._series)
    return reindexed


def to_datetime(implementation: str) -> Any:
    if implementation == "pandas":
        return get_pandas().to_datetime
    if implementation == "modin":
        return get_modin().to_datetime
    if implementation == "cudf":
        return get_cudf().to_datetime
    raise AssertionError
