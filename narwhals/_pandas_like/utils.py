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
            return item(other._series)
        if other._series.index is not index and not (other._series.index == index).all():
            return other._series.set_axis(index, axis=0)
        return other._series
    raise AssertionError("Please report a bug")


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


def parse_into_expr(implementation: str, into_expr: IntoPandasExpr) -> PandasExpr:
    """Parse `into_expr` as an expression.

    For example, in Polars, we can do both `df.select('a')` and `df.select(pl.col('a'))`.
    We do the same in Narwhals:

    - if `into_expr` is already an expression, just return it
    - if it's a Series, then convert it to an expression
    - if it's a numpy array, then convert it to a Series and then to an expression
    - if it's a string, then convert it to an expression
    - else, raise
    """
    from narwhals._pandas_like.expr import PandasExpr
    from narwhals._pandas_like.namespace import PandasNamespace
    from narwhals._pandas_like.series import PandasSeries

    plx = PandasNamespace(implementation=implementation)

    if isinstance(into_expr, PandasExpr):
        return into_expr
    if isinstance(into_expr, PandasSeries):
        return plx._create_expr_from_series(into_expr)
    if isinstance(into_expr, str):
        return plx.col(into_expr)
    if (np := get_numpy()) is not None and isinstance(into_expr, np.ndarray):
        series = create_native_series(into_expr, implementation=implementation)
        return plx._create_expr_from_series(series)
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
    *exprs: IntoPandasExpr | Iterable[IntoPandasExpr],
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
    from narwhals._pandas_like.expr import PandasExpr
    from narwhals._pandas_like.namespace import PandasNamespace

    plx = PandasNamespace(implementation=expr._implementation)

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
            assert [s._series.name for s in out] == expr._output_names
        return out

    # Try tracking root and output names by combining them from all
    # expressions appearing in args and kwargs. If any anonymous
    # expression appears (e.g. nw.all()), then give up on tracking root names
    # and just set it to None.
    root_names = copy(expr._root_names)
    output_names = expr._output_names
    for arg in list(args) + list(kwargs.values()):
        if root_names is not None and isinstance(arg, PandasExpr):
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


def item(s: Any) -> Any:
    # cuDF doesn't have Series.item().
    if len(s) != 1:
        msg = "Can only convert a Series of length 1 to a scalar"  # pragma: no cover
        raise AssertionError(msg)
    return s.iloc[0]


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


def series_from_iterable(
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


def translate_dtype(dtype: Any) -> DType:
    from narwhals import dtypes

    if dtype in ("int64", "Int64", "Int64[pyarrow]"):
        return dtypes.Int64()
    if dtype in ("int32", "Int32", "Int32[pyarrow]"):
        return dtypes.Int32()
    if dtype in ("int16", "Int16", "Int16[pyarrow]"):
        return dtypes.Int16()
    if dtype in ("int8", "Int8", "Int8[pyarrow]"):
        return dtypes.Int8()
    if dtype in ("uint64", "UInt64", "UInt64[pyarrow]"):
        return dtypes.UInt64()
    if dtype in ("uint32", "UInt32", "UInt32[pyarrow]"):
        return dtypes.UInt32()
    if dtype in ("uint16", "UInt16", "UInt16[pyarrow]"):
        return dtypes.UInt16()
    if dtype in ("uint8", "UInt8", "UInt8[pyarrow]"):
        return dtypes.UInt8()
    if dtype in ("float64", "Float64", "Float64[pyarrow]"):
        return dtypes.Float64()
    if dtype in ("float32", "Float32", "Float32[pyarrow]"):
        return dtypes.Float32()
    if dtype in ("string", "string[python]", "string[pyarrow]"):
        return dtypes.String()
    if dtype in ("bool", "boolean", "boolean[pyarrow]"):
        return dtypes.Boolean()
    if dtype in ("category",):
        return dtypes.Categorical()
    if str(dtype).startswith("datetime64"):
        # todo: different time units and time zones
        return dtypes.Datetime()
    if str(dtype).startswith("timestamp["):
        # pyarrow-backed datetime
        # todo: different time units and time zones
        return dtypes.Datetime()
    if dtype == "object":
        return dtypes.String()
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


def reverse_translate_dtype(  # noqa: PLR0915
    dtype: DType | type[DType], starting_dtype: Any, implementation: str
) -> Any:
    from narwhals import dtypes

    dtype_backend = get_dtype_backend(starting_dtype, implementation)
    if isinstance_or_issubclass(dtype, dtypes.Float64):
        if dtype_backend == "pyarrow-nullable":
            return "Float64[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "Float64"
        else:
            return "float64"
    if isinstance_or_issubclass(dtype, dtypes.Float32):
        if dtype_backend == "pyarrow-nullable":
            return "Float32[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "Float32"
        else:
            return "float32"
    if isinstance_or_issubclass(dtype, dtypes.Int64):
        if dtype_backend == "pyarrow-nullable":
            return "Int64[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "Int64"
        else:
            return "int64"
    if isinstance_or_issubclass(dtype, dtypes.Int32):
        if dtype_backend == "pyarrow-nullable":
            return "Int32[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "Int32"
        else:
            return "int32"
    if isinstance_or_issubclass(dtype, dtypes.Int16):
        if dtype_backend == "pyarrow-nullable":
            return "Int16[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "Int16"
        else:
            return "int16"
    if isinstance_or_issubclass(dtype, dtypes.Int8):
        if dtype_backend == "pyarrow-nullable":
            return "Int8[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "Int8"
        else:
            return "int8"
    if isinstance_or_issubclass(dtype, dtypes.UInt64):
        if dtype_backend == "pyarrow-nullable":
            return "UInt64[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "UInt64"
        else:
            return "uint64"
    if isinstance_or_issubclass(dtype, dtypes.UInt32):
        if dtype_backend == "pyarrow-nullable":
            return "UInt32[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "UInt32"
        else:
            return "uint32"
    if isinstance_or_issubclass(dtype, dtypes.UInt16):
        if dtype_backend == "pyarrow-nullable":
            return "UInt16[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "UInt16"
        else:
            return "uint16"
    if isinstance_or_issubclass(dtype, dtypes.UInt8):
        if dtype_backend == "pyarrow-nullable":
            return "UInt8[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "UInt8"
        else:
            return "uint8"
    if isinstance_or_issubclass(dtype, dtypes.String):
        if dtype_backend == "pyarrow-nullable":
            return "string[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "string"
        else:
            return str
    if isinstance_or_issubclass(dtype, dtypes.Boolean):
        if dtype_backend == "pyarrow-nullable":
            return "boolean[pyarrow]"
        if dtype_backend == "pandas-nullable":
            return "boolean"
        else:
            return "bool"
    if isinstance_or_issubclass(dtype, dtypes.Categorical):
        # todo: is there no pyarrow-backed categorical?
        # or at least, convert_dtypes(dtype_backend='pyarrow') doesn't
        # convert to it?
        return "category"
    if isinstance_or_issubclass(dtype, dtypes.Datetime):
        # todo: different time units and time zones
        if dtype_backend == "pyarrow-nullable":
            return "timestamp[ns][pyarrow]"
        return "datetime64[ns]"
    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def validate_indices(series: list[PandasSeries]) -> list[Any]:
    idx = series[0]._series.index
    reindexed = [series[0]._series]
    for s in series[1:]:
        if s._series.index is not idx and not (s._series.index == idx).all():
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
