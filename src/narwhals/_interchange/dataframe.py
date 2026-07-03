from __future__ import annotations

import enum
import sys
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final, NoReturn, Protocol

from narwhals._utils import Implementation, Version, _hasattr_static, parse_version
from narwhals.dependencies import (
    IMPORT_HOOKS as PANDAS_IMPORT_HOOKS,
    get_cudf,
    get_dask_dataframe,
    get_duckdb,
    get_ibis,
    get_modin,
    get_pandas,
    get_polars,
    get_pyarrow,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd
    import pyarrow as pa
    from typing_extensions import Self, TypeIs

    from narwhals.dtypes import DType


class DataFrameLike(Protocol):
    def __dataframe__(self, *args: Any, **kwargs: Any) -> Any: ...


class DtypeKind(enum.IntEnum):
    # https://data-apis.org/dataframe-protocol/latest/API.html
    INT = 0
    UINT = 1
    FLOAT = 2
    BOOL = 20
    STRING = 21  # UTF-8
    DATETIME = 22
    CATEGORICAL = 23


def map_interchange_dtype_to_narwhals_dtype(  # noqa: C901, PLR0911, PLR0912
    interchange_dtype: tuple[DtypeKind, int, Any, Any],
) -> DType:
    dtypes = Version.V1.dtypes
    if interchange_dtype[0] == DtypeKind.INT:
        if interchange_dtype[1] == 64:
            return dtypes.Int64()
        if interchange_dtype[1] == 32:
            return dtypes.Int32()
        if interchange_dtype[1] == 16:
            return dtypes.Int16()
        if interchange_dtype[1] == 8:
            return dtypes.Int8()
        msg = "Invalid bit width for INT"  # pragma: no cover
        raise AssertionError(msg)
    if interchange_dtype[0] == DtypeKind.UINT:
        if interchange_dtype[1] == 64:
            return dtypes.UInt64()
        if interchange_dtype[1] == 32:
            return dtypes.UInt32()
        if interchange_dtype[1] == 16:
            return dtypes.UInt16()
        if interchange_dtype[1] == 8:
            return dtypes.UInt8()
        msg = "Invalid bit width for UINT"  # pragma: no cover
        raise AssertionError(msg)
    if interchange_dtype[0] == DtypeKind.FLOAT:
        if interchange_dtype[1] == 64:
            return dtypes.Float64()
        if interchange_dtype[1] == 32:
            return dtypes.Float32()
        if interchange_dtype[1] == 16:
            return dtypes.Float16()
        msg = "Invalid bit width for FLOAT"  # pragma: no cover
        raise AssertionError(msg)
    if interchange_dtype[0] == DtypeKind.BOOL:
        return dtypes.Boolean()
    if interchange_dtype[0] == DtypeKind.STRING:
        return dtypes.String()
    if interchange_dtype[0] == DtypeKind.DATETIME:
        return dtypes.Datetime()
    if interchange_dtype[0] == DtypeKind.CATEGORICAL:  # pragma: no cover
        # upstream issue: https://github.com/ibis-project/ibis/issues/9570
        return dtypes.Categorical()
    msg = f"Invalid dtype, got: {interchange_dtype}"  # pragma: no cover
    raise AssertionError(msg)


def unsupported_error(method_name: str) -> NotImplementedError:
    msg = (
        f"{method_name!r} is not supported for interchange-level dataframes.\n\n"
        "Hint: you probably called `from_native` on an object which isn't fully "
        "supported by `narwhals.stable.v1`, yet implements `__dataframe__`."
    )
    return NotImplementedError(msg)


class _Interchange:
    _version: Final = Version.V1
    _implementation: Final = Implementation.UNKNOWN

    def __native_namespace__(self) -> NoReturn:
        kind = "series" if "Series" in type(self).__name__ else "dataframes"
        msg = f"Cannot access native namespace for interchange-level {kind} with unknown backend."
        raise NotImplementedError(msg)

    def __getattr__(self, attr: str) -> Any:
        raise unsupported_error(attr)


class InterchangeFrame(_Interchange):
    def __init__(self, df: DataFrameLike) -> None:
        self._dfi: Any = df.__dataframe__()

    def __narwhals_dataframe__(self) -> Self:
        return self

    def get_column(self, name: str) -> InterchangeSeries:
        return InterchangeSeries(self._dfi.get_column_by_name(name))

    def to_pandas(self) -> pd.DataFrame:
        import pandas as pd  # ignore-banned-import()

        if parse_version(pd) < (1, 5, 0):  # pragma: no cover
            msg = (
                "Conversion to pandas is achieved via interchange protocol which requires"
                f" 'pandas>=1.5.0' to be installed, found {pd.__version__}"
            )
            raise NotImplementedError(msg)
        return pd.api.interchange.from_dataframe(self._dfi)

    def to_arrow(self) -> pa.Table:
        from pyarrow.interchange.from_dataframe import (  # ignore-banned-import()
            from_dataframe,
        )

        return from_dataframe(self._dfi)

    @property
    def schema(self) -> dict[str, DType]:
        return {
            column_name: map_interchange_dtype_to_narwhals_dtype(
                self._dfi.get_column_by_name(column_name).dtype
            )
            for column_name in self._dfi.column_names()
        }

    @property
    def columns(self) -> list[str]:
        return list(self._dfi.column_names())

    def simple_select(self, *column_names: str) -> Self:
        frame = self._dfi.select_columns_by_name(list(column_names))
        return self.__class__(frame._df)


class InterchangeSeries(_Interchange):
    def __init__(self, df: Any) -> None:
        self._native_series = df

    def __narwhals_series__(self) -> Self:
        return self

    @property
    def dtype(self) -> DType:
        return map_interchange_dtype_to_narwhals_dtype(self._native_series.dtype)

    @property
    def native(self) -> Any:
        return self._native_series


def should_interchange(obj: object) -> TypeIs[DataFrameLike]:
    """Return True if we'd use interchange/metadata-level support for `obj`."""
    return _should_interchange(type(obj))  # type: ignore[arg-type]


def _iter_exclude_interchange() -> Iterator[type[DataFrameLike]]:
    """Yield classes that we want to **avoid** using with `__dataframe__`.

    We *could* use them with interchange-level support - but we have a better option at home.

    Sadly this type is [not yet](https://www.youtube.com/watch?v=xUsPD7iwETY) spellable:

        (DataFrameLike & (pl.DataFrame | pd.DataFrame | pa.Table | dd.DataFrame | mpd.DataFrame | cudf.DataFrame)
        #              |             union
        #         intersection

    We gathers all the types (that are already imported) which fit the the right-hand-side,
    as we need to refine from the set produced by checking `__dataframe__`.
    """
    has_top_level_df = (get_polars, get_pandas, get_dask_dataframe, get_modin, get_cudf)
    for get_module in has_top_level_df:
        if module := get_module():
            yield module.DataFrame
    if pa := get_pyarrow():
        yield pa.Table
    for module_name in PANDAS_IMPORT_HOOKS:  # pragma: no cover
        if module := sys.modules.get(module_name):
            yield module.pandas.DataFrame


@lru_cache(64)
def _should_interchange(tp_native: type[Any]) -> TypeIs[type[DataFrameLike]]:
    if _hasattr_static(tp_native, "__dataframe__"):
        # Ensure we use `__dataframe__` when it is the **only** option we have
        exclude = tuple(_iter_exclude_interchange())
        return (not exclude) or (not issubclass(tp_native, exclude))
    include = (duckdb.DuckDBPyRelation,) if (duckdb := get_duckdb()) else ()
    # Gracefully handle `ibis` removing `__dataframe__` support in the future (we don't use it anyway)
    include = (*include, ibis.Table) if (ibis := get_ibis()) else include
    return bool(include) and issubclass(tp_native, include)
