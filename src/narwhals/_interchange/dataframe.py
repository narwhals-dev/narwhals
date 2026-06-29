from __future__ import annotations

import enum
import sys
from functools import lru_cache
from inspect import getattr_static
from typing import TYPE_CHECKING, Any, Final, NoReturn, Protocol, TypeVar

from narwhals import dependencies as deps
from narwhals._utils import Implementation, Version, _hasattr_static, parse_version

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from types import ModuleType

    import pandas as pd
    import pyarrow as pa
    from typing_extensions import Self, TypeIs

    from narwhals._interchange.series import InterchangeSeries
    from narwhals.dtypes import DType
    from narwhals.stable.v1 import DataFrame as DataFrameV1


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


Original_co = TypeVar("Original_co", bound=DataFrameLike, covariant=True)


class Column(Protocol):
    @property
    def dtype(self) -> tuple[DtypeKind, int, Any, Any] | Any: ...


class RecoverableColumn(Column, Protocol[Original_co]):  # type: ignore[misc]
    _version: Version = Version.V1
    _native_series: Original_co

    def __narwhals_series__(self) -> Self:
        return self


class InterchangeSeriesV1(RecoverableColumn[Original_co], Protocol[Original_co]):  # type: ignore[misc]
    _implementation: Implementation

    @property
    def dtype(self) -> DType: ...  # ?

    def __native_namespace__(self) -> ModuleType:
        return self._implementation.to_native_namespace()


class Frame(Protocol):
    def __dataframe__(self, *_: Any, **__: Any) -> Self:  # pragma: no cover
        return self

    def column_names(self) -> Iterable[str]: ...
    def get_column_by_name(self, name: str, /) -> Column: ...
    def select_columns_by_name(self, names: Sequence[str], /) -> Self: ...


class RecoverableFrame(Frame, Protocol[Original_co]):
    _version: Version = Version.V1

    @property
    def _df(self) -> Original_co: ...

    """Allow for recovering original object.

    See https://github.com/data-apis/dataframe-api/issues/360.
    """

    def __narwhals_dataframe__(self) -> Self:
        return self


DFI_METHODS = (
    "column_names",
    "__dataframe__",
    "get_chunks",
    "get_column",
    "get_column_by_name",
    "get_columns",
    "metadata",
    "num_chunks",
    "num_columns",
    "num_rows",
    "select_columns",
    "select_columns_by_name",
)

SENTINEL = object()


class WrapsInterchangeFrame(Protocol):
    _dfi: Frame

    def __getattr__(self, attr: str) -> Any:
        if (
            attr in DFI_METHODS
            and (func := getattr_static(self._dfi, attr, SENTINEL)) is not SENTINEL
        ):
            return func
        msg = (
            f"Attribute {attr} is not supported for interchange-level dataframes.\n\n"
            "Hint: you probably called `from_native` on an object which isn't fully "
            "supported by `narwhals.stable.v1`, yet implements `__dataframe__`."
        )
        raise NotImplementedError(msg)


class InterchangeFrameV1(
    WrapsInterchangeFrame, RecoverableFrame[Original_co], Protocol[Original_co]
):
    _implementation: Implementation

    @property
    def schema(self) -> dict[str, DType]: ...
    @property
    def columns(self) -> list[str]:
        return list(self.column_names())

    @property
    def _native_frame(self) -> Original_co:
        return self._df

    def to_pandas(self) -> pd.DataFrame: ...
    def to_arrow(self) -> pa.Table: ...
    def get_column(self, name: str, /) -> InterchangeSeriesV1[Original_co]: ...
    def get_column_by_name(
        self, name: str, /
    ) -> InterchangeSeriesV1[Original_co]:  # pragma: no cover
        return self.get_column(name)

    def collect_schema(self) -> dict[str, DType]:
        return self.schema

    def __native_namespace__(self) -> ModuleType:
        return self._implementation.to_native_namespace()

    def simple_select(self, *column_names: str) -> Self:
        return self.select_columns_by_name(column_names)

    def to_narwhals(self) -> DataFrameV1[Any]:  # pragma: no cover
        from narwhals.stable.v1 import DataFrame as DataFrameV1

        return DataFrameV1(self)  # type: ignore[no-any-return]


# TODO @dangotbanned: Review what is going on here
# - roll in the protocol stuff
# - integrate duckdb
class InterchangeFrame(WrapsInterchangeFrame):
    _version = Version.V1
    _implementation: Final = Implementation.UNKNOWN

    def __init__(self, df: DataFrameLike) -> None:
        self._dfi = df.__dataframe__()

    def __narwhals_dataframe__(self) -> Self:
        return self

    def __native_namespace__(self) -> NoReturn:
        msg = (
            "Cannot access native namespace for interchange-level dataframes with unknown backend."
            "If you would like to see this kind of object supported in Narwhals, please "
            "open a feature request at https://github.com/narwhals-dev/narwhals/issues."
        )
        raise NotImplementedError(msg)

    def get_column(self, name: str) -> InterchangeSeries:
        from narwhals._interchange.series import InterchangeSeries

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
        if not hasattr(frame, "_df"):  # pragma: no cover
            msg = (
                "Expected interchange object to implement `_df` property to allow for recovering original object.\n"
                "See https://github.com/data-apis/dataframe-api/issues/360."
            )
            raise NotImplementedError(msg)
        return self.__class__(frame._df)  # pyright: ignore[reportAttributeAccessIssue]

    def select(self, *exprs: str) -> Self:  # pragma: no cover
        msg = (
            "`select`-ing not by name is not supported for interchange-only level.\n\n"
            "If you would like to see this kind of object better supported in "
            "Narwhals, please open a feature request "
            "at https://github.com/narwhals-dev/narwhals/issues."
        )
        raise NotImplementedError(msg)


def supports_dataframe_interchange(obj: Any) -> TypeIs[DataFrameLike]:
    return _hasattr_static(obj, "__dataframe__")


def should_interchange(obj: object) -> TypeIs[DataFrameLike]:
    return _should_interchange(type(obj))  # type: ignore[arg-type]


_HAS_TOP_LEVEL_DF = (
    deps.get_polars,
    deps.get_pandas,
    deps.get_dask_dataframe,
    deps.get_modin,
)


# TODO @dangotbanned: ~~cudf~~, sqlframe?, pyspark?, pyspark-connect?
@lru_cache(64)
def _should_interchange(tp_native: type[Any]) -> TypeIs[type[DataFrameLike]]:
    if not supports_dataframe_interchange(tp_native):
        return (duckdb := deps.get_duckdb()) and issubclass(
            tp_native, duckdb.DuckDBPyRelation
        )
    exclude = tuple(mod.DataFrame for get in _HAS_TOP_LEVEL_DF if (mod := get()))
    exclude = (*exclude, pa.Table) if (pa := deps.get_pyarrow()) else exclude
    hooks = (
        mod.pandas.DataFrame
        for name in deps.IMPORT_HOOKS
        if (mod := sys.modules.get(name))
    )
    return not (exclude := (*exclude, *hooks)) or not issubclass(tp_native, exclude)
