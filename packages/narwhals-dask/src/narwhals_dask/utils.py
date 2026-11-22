from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals.dependencies import get_pyarrow
from narwhals.utils import Version
from narwhals_dask._vendored import isinstance_or_issubclass, select_columns_by_name

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import dask.dataframe as dd
    import dask.dataframe.dask_expr as dx

    from narwhals import Schema
    from narwhals.dtypes import DType
    from narwhals.typing import IntoDType
    from narwhals_dask.dataframe import DaskLazyFrame, Incomplete
    from narwhals_dask.expr import DaskExpr
else:
    try:
        import dask.dataframe.dask_expr as dx
    except ModuleNotFoundError:
        import dask_expr as dx


def evaluate_exprs(df: DaskLazyFrame, /, *exprs: DaskExpr) -> list[tuple[str, dx.Series]]:
    native_results: list[tuple[str, dx.Series]] = []
    for expr in exprs:
        native_series_list = expr(df)
        aliases = expr._evaluate_aliases(df)
        if len(aliases) != len(native_series_list):  # pragma: no cover
            msg = f"Internal error: got aliases {aliases}, but only got {len(native_series_list)} results"
            raise AssertionError(msg)
        native_results.extend(zip(aliases, native_series_list, strict=False))
    return native_results


def align_series_full_broadcast(
    df: DaskLazyFrame, *series: dx.Series | object
) -> Sequence[dx.Series]:
    return [
        s if isinstance(s, dx.Series) else df._native_frame.assign(_tmp=s)["_tmp"]
        for s in series
    ]  # pyright: ignore[reportReturnType]


def add_row_index(frame: dd.DataFrame, name: str) -> dd.DataFrame:
    original_cols = frame.columns
    df: Incomplete = frame.assign(**{name: 1})
    return select_columns_by_name(
        df.assign(**{name: df[name].cumsum(method="blelloch") - 1}),
        [name, *original_cols],
    )


def validate_comparand(lhs: dx.Series, rhs: dx.Series) -> None:
    if not dx.expr.are_co_aligned(lhs._expr, rhs._expr):  # pragma: no cover
        # are_co_aligned is a method which cheaply checks if two Dask expressions
        # have the same index, and therefore don't require index alignment.
        # If someone only operates on a Dask DataFrame via expressions, then this
        # should always be the case: expression outputs (by definition) all come from the
        # same input dataframe, and Dask Series does not have any operations which
        # change the index. Nonetheless, we perform this safety check anyway.

        # However, we still need to carefully vet which methods we support for Dask, to
        # avoid issues where `are_co_aligned` doesn't do what we want it to do:
        # https://github.com/dask/dask-expr/issues/1112.
        msg = "Objects are not co-aligned, so this operation is not supported for Dask backend"
        raise RuntimeError(msg)


dtypes = Version.MAIN.dtypes
dtypes_v1 = Version.V1.dtypes
NW_TO_DASK_DTYPES: Mapping[type[DType], str] = {
    dtypes.Float64: "float64",
    dtypes.Float32: "float32",
    dtypes.Float16: "float16",
    dtypes.Boolean: "bool",
    dtypes.Categorical: "category",
    dtypes.Date: "date32[day][pyarrow]",
    dtypes.Int8: "int8",
    dtypes.Int16: "int16",
    dtypes.Int32: "int32",
    dtypes.Int64: "int64",
    dtypes.UInt8: "uint8",
    dtypes.UInt16: "uint16",
    dtypes.UInt32: "uint32",
    dtypes.UInt64: "uint64",
    dtypes.Datetime: "datetime64[us]",
    dtypes.Duration: "timedelta64[ns]",
    dtypes_v1.Datetime: "datetime64[us]",
    dtypes_v1.Duration: "timedelta64[ns]",
}
UNSUPPORTED_DTYPES = (
    dtypes.Struct,
    dtypes.Array,
    dtypes.Map,
    dtypes.Time,
    dtypes.Binary,
    dtypes.Decimal,
)


def narwhals_to_native_dtype(dtype: IntoDType, version: Version) -> Any:
    dtypes = version.dtypes
    base_type = dtype.base_type()
    if dask_type := NW_TO_DASK_DTYPES.get(base_type):
        return dask_type
    if isinstance_or_issubclass(dtype, dtypes.String):
        # NOTE: `dask[dataframe]>=2024.8` requires `pandas>=2`.
        return "string[pyarrow]" if get_pyarrow() else "string[python]"
    if isinstance_or_issubclass(dtype, dtypes.Enum):
        if version is Version.V1:
            msg = "Converting to Enum is not supported in narwhals.stable.v1"
            raise NotImplementedError(msg)
        if isinstance(dtype, dtypes.Enum):
            import pandas as pd

            # NOTE: `pandas-stubs.core.dtypes.dtypes.CategoricalDtype.categories` is too narrow
            # Should be one of the `ListLike*` types
            # https://github.com/pandas-dev/pandas-stubs/blob/8434bde95460b996323cc8c0fea7b0a8bb00ea26/pandas-stubs/_typing.pyi#L497-L505
            return pd.CategoricalDtype(dtype.categories, ordered=True)  # type: ignore[arg-type]
        msg = "Can not cast / initialize Enum without categories present"
        raise ValueError(msg)
    if isinstance_or_issubclass(dtype, dtypes.List):
        import pandas as pd

        arrow_schema = _schema_class(version)({"": dtype}).to_arrow()
        return pd.ArrowDtype(arrow_schema.field("").type)
    if issubclass(base_type, UNSUPPORTED_DTYPES):  # pragma: no cover
        msg = f"Converting to {base_type.__name__} dtype is not supported for Dask."
        raise NotImplementedError(msg)
    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def _schema_class(version: Version) -> type[Schema]:
    if version is Version.V1:
        from narwhals.stable.v1 import Schema as SchemaV1

        return SchemaV1
    if version is Version.V2:
        from narwhals.stable.v2 import Schema as SchemaV2

        return SchemaV2
    from narwhals import Schema

    return Schema


def native_to_narwhals_dtype(native_dtype: Any, version: Version) -> DType:
    if str(native_dtype) == "object":
        # Per conversations with their maintainers, Dask doesn't support arbitrary
        # objects, so we can just return String.
        return version.dtypes.String()
    return _schema_class(version).from_pandas_like({"": native_dtype})[""]
