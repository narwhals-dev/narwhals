from __future__ import annotations

import os
import re
from datetime import timezone
from enum import Enum
from enum import auto
from functools import wraps
from importlib.util import find_spec
from inspect import getattr_static
from inspect import getdoc
from secrets import token_hex
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Container
from typing import Iterable
from typing import Literal
from typing import Protocol
from typing import Sequence
from typing import TypeVar
from typing import Union
from typing import cast
from typing import overload
from warnings import warn

from narwhals.dependencies import get_cudf
from narwhals.dependencies import get_dask
from narwhals.dependencies import get_dask_dataframe
from narwhals.dependencies import get_duckdb
from narwhals.dependencies import get_ibis
from narwhals.dependencies import get_modin
from narwhals.dependencies import get_pandas
from narwhals.dependencies import get_polars
from narwhals.dependencies import get_pyarrow
from narwhals.dependencies import get_pyspark
from narwhals.dependencies import get_pyspark_connect
from narwhals.dependencies import get_pyspark_sql
from narwhals.dependencies import get_sqlframe
from narwhals.dependencies import is_cudf_series
from narwhals.dependencies import is_modin_series
from narwhals.dependencies import is_narwhals_series
from narwhals.dependencies import is_narwhals_series_int
from narwhals.dependencies import is_numpy_array_1d
from narwhals.dependencies import is_numpy_array_1d_int
from narwhals.dependencies import is_pandas_dataframe
from narwhals.dependencies import is_pandas_like_dataframe
from narwhals.dependencies import is_pandas_like_series
from narwhals.dependencies import is_pandas_series
from narwhals.dependencies import is_polars_series
from narwhals.dependencies import is_pyarrow_chunked_array
from narwhals.exceptions import ColumnNotFoundError
from narwhals.exceptions import DuplicateError
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from types import ModuleType
    from typing import AbstractSet as Set

    import pandas as pd
    import pyarrow as pa
    from typing_extensions import Concatenate
    from typing_extensions import LiteralString
    from typing_extensions import ParamSpec
    from typing_extensions import Self
    from typing_extensions import TypeAlias
    from typing_extensions import TypeIs

    from narwhals._compliant import CompliantExpr
    from narwhals._compliant import CompliantExprT
    from narwhals._compliant import CompliantFrameT
    from narwhals._compliant import CompliantSeriesOrNativeExprT_co
    from narwhals._compliant import CompliantSeriesT
    from narwhals._compliant import NativeFrameT_co
    from narwhals._compliant import NativeSeriesT_co
    from narwhals._compliant.typing import EvalNames
    from narwhals._namespace import EagerAllowedImplementation
    from narwhals._namespace import Namespace
    from narwhals._translate import ArrowStreamExportable
    from narwhals._translate import IntoArrowTable
    from narwhals.dataframe import DataFrame
    from narwhals.dataframe import LazyFrame
    from narwhals.dtypes import DType
    from narwhals.series import Series
    from narwhals.typing import CompliantDataFrame
    from narwhals.typing import CompliantLazyFrame
    from narwhals.typing import CompliantSeries
    from narwhals.typing import DataFrameLike
    from narwhals.typing import DTypes
    from narwhals.typing import IntoSeriesT
    from narwhals.typing import MultiIndexSelector
    from narwhals.typing import SingleIndexSelector
    from narwhals.typing import SizedMultiIndexSelector
    from narwhals.typing import SizeUnit
    from narwhals.typing import SupportsNativeNamespace
    from narwhals.typing import TimeUnit
    from narwhals.typing import _1DArray
    from narwhals.typing import _SliceIndex
    from narwhals.typing import _SliceName
    from narwhals.typing import _SliceNone

    FrameOrSeriesT = TypeVar(
        "FrameOrSeriesT", bound=Union[LazyFrame[Any], DataFrame[Any], Series[Any]]
    )
    _T = TypeVar("_T")
    _T1 = TypeVar("_T1")
    _T2 = TypeVar("_T2")
    _T3 = TypeVar("_T3")
    _Fn = TypeVar("_Fn", bound="Callable[..., Any]")
    P = ParamSpec("P")
    R = TypeVar("R")
    R1 = TypeVar("R1")
    R2 = TypeVar("R2")

    class _SupportsVersion(Protocol):
        __version__: str

    class _SupportsGet(Protocol):  # noqa: PYI046
        def __get__(self, instance: Any, owner: Any | None = None, /) -> Any: ...

    class _StoresImplementation(Protocol):
        _implementation: Implementation
        """Implementation of native object (pandas, Polars, PyArrow, ...)."""

    class _StoresBackendVersion(Protocol):
        _backend_version: tuple[int, ...]
        """Version tuple for a native package."""

    class _StoresVersion(Protocol):
        _version: Version
        """Narwhals API version (V1 or MAIN)."""

    class _LimitedContext(_StoresBackendVersion, _StoresVersion, Protocol):
        """Provides 2 attributes.

        - `_backend_version`
        - `_version`
        """

    class _FullContext(_StoresImplementation, _LimitedContext, Protocol):
        """Provides 3 attributes.

        - `_implementation`
        - `_backend_version`
        - `_version`
        """

    class _StoresColumns(Protocol):
        @property
        def columns(self) -> Sequence[str]: ...


NativeT_co = TypeVar("NativeT_co", covariant=True)
CompliantT_co = TypeVar("CompliantT_co", covariant=True)
_ContextT = TypeVar("_ContextT", bound="_FullContext")
_Method: TypeAlias = "Callable[Concatenate[_ContextT, P], R]"
_Constructor: TypeAlias = "Callable[Concatenate[_T, P], R2]"


class _StoresNative(Protocol[NativeT_co]):  # noqa: PYI046
    """Provides access to a native object.

    Native objects have types like:

    >>> from pandas import Series
    >>> from pyarrow import Table
    """

    @property
    def native(self) -> NativeT_co:
        """Return the native object."""
        ...


class _StoresCompliant(Protocol[CompliantT_co]):  # noqa: PYI046
    """Provides access to a compliant object.

    Compliant objects have types like:

    >>> from narwhals._pandas_like.series import PandasLikeSeries
    >>> from narwhals._arrow.dataframe import ArrowDataFrame
    """

    @property
    def compliant(self) -> CompliantT_co:
        """Return the compliant object."""
        ...


class Version(Enum):
    V1 = auto()
    MAIN = auto()

    @property
    def namespace(self) -> type[Namespace[Any]]:
        if self is Version.MAIN:
            from narwhals._namespace import Namespace

            return Namespace
        from narwhals.stable.v1._namespace import Namespace

        return Namespace

    @property
    def dtypes(self) -> DTypes:
        if self is Version.MAIN:
            from narwhals import dtypes

            return dtypes
        from narwhals.stable.v1 import dtypes as v1_dtypes

        return v1_dtypes


class Implementation(Enum):
    """Implementation of native object (pandas, Polars, PyArrow, ...)."""

    PANDAS = auto()
    """Pandas implementation."""
    MODIN = auto()
    """Modin implementation."""
    CUDF = auto()
    """cuDF implementation."""
    PYARROW = auto()
    """PyArrow implementation."""
    PYSPARK = auto()
    """PySpark implementation."""
    POLARS = auto()
    """Polars implementation."""
    DASK = auto()
    """Dask implementation."""
    DUCKDB = auto()
    """DuckDB implementation."""
    IBIS = auto()
    """Ibis implementation."""
    SQLFRAME = auto()
    """SQLFrame implementation."""
    PYSPARK_CONNECT = auto()
    """PySpark Connect implementation."""

    UNKNOWN = auto()
    """Unknown implementation."""

    @classmethod
    def from_native_namespace(
        cls: type[Self], native_namespace: ModuleType
    ) -> Implementation:  # pragma: no cover
        """Instantiate Implementation object from a native namespace module.

        Arguments:
            native_namespace: Native namespace.

        Returns:
            Implementation.
        """
        mapping = {
            get_pandas(): Implementation.PANDAS,
            get_modin(): Implementation.MODIN,
            get_cudf(): Implementation.CUDF,
            get_pyarrow(): Implementation.PYARROW,
            get_pyspark_sql(): Implementation.PYSPARK,
            get_polars(): Implementation.POLARS,
            get_dask_dataframe(): Implementation.DASK,
            get_duckdb(): Implementation.DUCKDB,
            get_ibis(): Implementation.IBIS,
            get_sqlframe(): Implementation.SQLFRAME,
            get_pyspark_connect(): Implementation.PYSPARK_CONNECT,
        }
        return mapping.get(native_namespace, Implementation.UNKNOWN)

    @classmethod
    def from_string(
        cls: type[Self], backend_name: str
    ) -> Implementation:  # pragma: no cover
        """Instantiate Implementation object from a native namespace module.

        Arguments:
            backend_name: Name of backend, expressed as string.

        Returns:
            Implementation.
        """
        mapping = {
            "pandas": Implementation.PANDAS,
            "modin": Implementation.MODIN,
            "cudf": Implementation.CUDF,
            "pyarrow": Implementation.PYARROW,
            "pyspark": Implementation.PYSPARK,
            "polars": Implementation.POLARS,
            "dask": Implementation.DASK,
            "duckdb": Implementation.DUCKDB,
            "ibis": Implementation.IBIS,
            "sqlframe": Implementation.SQLFRAME,
            "pyspark_connect": Implementation.PYSPARK_CONNECT,
        }
        return mapping.get(backend_name, Implementation.UNKNOWN)

    @classmethod
    def from_backend(
        cls: type[Self], backend: str | Implementation | ModuleType
    ) -> Implementation:
        """Instantiate from native namespace module, string, or Implementation.

        Arguments:
            backend: Backend to instantiate Implementation from.

        Returns:
            Implementation.
        """
        return (
            cls.from_string(backend)
            if isinstance(backend, str)
            else backend
            if isinstance(backend, Implementation)
            else cls.from_native_namespace(backend)
        )

    def to_native_namespace(self) -> ModuleType:
        """Return the native namespace module corresponding to Implementation.

        Returns:
            Native module.
        """
        if self is Implementation.PANDAS:
            import pandas as pd  # ignore-banned-import

            return pd
        if self is Implementation.MODIN:
            import modin.pandas

            return modin.pandas
        if self is Implementation.CUDF:  # pragma: no cover
            import cudf  # ignore-banned-import

            return cudf
        if self is Implementation.PYARROW:
            import pyarrow as pa  # ignore-banned-import

            return pa
        if self is Implementation.PYSPARK:  # pragma: no cover
            import pyspark.sql

            return pyspark.sql
        if self is Implementation.POLARS:
            import polars as pl  # ignore-banned-import

            return pl
        if self is Implementation.DASK:
            import dask.dataframe  # ignore-banned-import

            return dask.dataframe

        if self is Implementation.DUCKDB:
            import duckdb  # ignore-banned-import

            return duckdb

        if self is Implementation.SQLFRAME:
            import sqlframe  # ignore-banned-import

            return sqlframe

        if self is Implementation.PYSPARK_CONNECT:  # pragma: no cover
            import pyspark.sql  # ignore-banned-import

            return pyspark.sql

        msg = "Not supported Implementation"  # pragma: no cover
        raise AssertionError(msg)

    def is_pandas(self) -> bool:
        """Return whether implementation is pandas.

        Returns:
            Boolean.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.implementation.is_pandas()
            True
        """
        return self is Implementation.PANDAS

    def is_pandas_like(self) -> bool:
        """Return whether implementation is pandas, Modin, or cuDF.

        Returns:
            Boolean.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.implementation.is_pandas_like()
            True
        """
        return self in {
            Implementation.PANDAS,
            Implementation.MODIN,
            Implementation.CUDF,
        }

    def is_spark_like(self) -> bool:
        """Return whether implementation is pyspark or sqlframe.

        Returns:
            Boolean.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.implementation.is_spark_like()
            False
        """
        return self in {
            Implementation.PYSPARK,
            Implementation.SQLFRAME,
            Implementation.PYSPARK_CONNECT,
        }

    def is_polars(self) -> bool:
        """Return whether implementation is Polars.

        Returns:
            Boolean.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.implementation.is_polars()
            True
        """
        return self is Implementation.POLARS

    def is_cudf(self) -> bool:
        """Return whether implementation is cuDF.

        Returns:
            Boolean.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.implementation.is_cudf()
            False
        """
        return self is Implementation.CUDF  # pragma: no cover

    def is_modin(self) -> bool:
        """Return whether implementation is Modin.

        Returns:
            Boolean.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.implementation.is_modin()
            False
        """
        return self is Implementation.MODIN  # pragma: no cover

    def is_pyspark(self) -> bool:
        """Return whether implementation is PySpark.

        Returns:
            Boolean.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.implementation.is_pyspark()
            False
        """
        return self is Implementation.PYSPARK  # pragma: no cover

    def is_pyspark_connect(self) -> bool:
        """Return whether implementation is PySpark.

        Returns:
            Boolean.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.implementation.is_pyspark_connect()
            False
        """
        return self is Implementation.PYSPARK_CONNECT  # pragma: no cover

    def is_pyarrow(self) -> bool:
        """Return whether implementation is PyArrow.

        Returns:
            Boolean.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.implementation.is_pyarrow()
            False
        """
        return self is Implementation.PYARROW  # pragma: no cover

    def is_dask(self) -> bool:
        """Return whether implementation is Dask.

        Returns:
            Boolean.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.implementation.is_dask()
            False
        """
        return self is Implementation.DASK  # pragma: no cover

    def is_duckdb(self) -> bool:
        """Return whether implementation is DuckDB.

        Returns:
            Boolean.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.implementation.is_duckdb()
            False
        """
        return self is Implementation.DUCKDB  # pragma: no cover

    def is_ibis(self) -> bool:
        """Return whether implementation is Ibis.

        Returns:
            Boolean.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.implementation.is_ibis()
            False
        """
        return self is Implementation.IBIS  # pragma: no cover

    def is_sqlframe(self) -> bool:
        """Return whether implementation is SQLFrame.

        Returns:
            Boolean.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.implementation.is_sqlframe()
            False
        """
        return self is Implementation.SQLFRAME  # pragma: no cover

    @property
    def _alias(self) -> LiteralString:
        """Friendly name for errors.

        Returns:
            String.
        """
        mapping: dict[Implementation, LiteralString] = {
            Implementation.PANDAS: "Pandas",
            Implementation.POLARS: "Polars",
            Implementation.DASK: "Dask",
            Implementation.IBIS: "Ibis",
            Implementation.MODIN: "Modin",
            Implementation.CUDF: "cuDF",
            Implementation.PYARROW: "PyArrow",
            Implementation.PYSPARK: "PySpark",
            Implementation.DUCKDB: "DuckDB",
            Implementation.SQLFRAME: "SQLFrame",
            Implementation.PYSPARK_CONNECT: "PySpark Connect",
        }
        return mapping[self]

    def _backend_version(self) -> tuple[int, ...]:
        native = self.to_native_namespace()
        into_version: Any
        if self not in {
            Implementation.PYSPARK,
            Implementation.PYSPARK_CONNECT,
            Implementation.DASK,
            Implementation.SQLFRAME,
        }:
            into_version = native
        elif self in {Implementation.PYSPARK, Implementation.PYSPARK_CONNECT}:
            into_version = get_pyspark()  # pragma: no cover
        elif self is Implementation.DASK:
            into_version = get_dask()
        else:
            import sqlframe._version

            into_version = sqlframe._version
        return parse_version(into_version)


MIN_VERSIONS: dict[Implementation, tuple[int, ...]] = {
    Implementation.PANDAS: (0, 25, 3),
    Implementation.MODIN: (0, 25, 3),
    Implementation.CUDF: (24, 10),
    Implementation.PYARROW: (11,),
    Implementation.PYSPARK: (3, 5),
    Implementation.PYSPARK_CONNECT: (3, 5),
    Implementation.POLARS: (0, 20, 3),
    Implementation.DASK: (2024, 8),
    Implementation.DUCKDB: (1,),
    Implementation.IBIS: (6,),
    Implementation.SQLFRAME: (3, 22, 0),
}


def validate_backend_version(
    implementation: Implementation, backend_version: tuple[int, ...]
) -> None:
    if backend_version < (min_version := MIN_VERSIONS[implementation]):
        msg = f"Minimum version of {implementation} supported by Narwhals is {min_version}, found: {backend_version}"
        raise ValueError(msg)


def remove_prefix(text: str, prefix: str) -> str:  # pragma: no cover
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def remove_suffix(text: str, suffix: str) -> str:  # pragma: no cover
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text  # pragma: no cover


def flatten(args: Any) -> list[Any]:
    return list(args[0] if (len(args) == 1 and _is_iterable(args[0])) else args)


def tupleify(arg: Any) -> Any:
    if not isinstance(arg, (list, tuple)):  # pragma: no cover
        return (arg,)
    return arg


def _is_iterable(arg: Any | Iterable[Any]) -> bool:
    from narwhals.series import Series

    if is_pandas_dataframe(arg) or is_pandas_series(arg):
        msg = f"Expected Narwhals class or scalar, got: {type(arg)}. Perhaps you forgot a `nw.from_native` somewhere?"
        raise TypeError(msg)
    if (pl := get_polars()) is not None and isinstance(
        arg, (pl.Series, pl.Expr, pl.DataFrame, pl.LazyFrame)
    ):
        msg = (
            f"Expected Narwhals class or scalar, got: {type(arg)}.\n\n"
            "Hint: Perhaps you\n"
            "- forgot a `nw.from_native` somewhere?\n"
            "- used `pl.col` instead of `nw.col`?"
        )
        raise TypeError(msg)

    return isinstance(arg, Iterable) and not isinstance(arg, (str, bytes, Series))


def parse_version(version: str | ModuleType | _SupportsVersion) -> tuple[int, ...]:
    """Simple version parser; split into a tuple of ints for comparison.

    Arguments:
        version: Version string, or object with one, to parse.

    Returns:
        Parsed version number.
    """
    # lifted from Polars
    # [marco]: Take care of DuckDB pre-releases which end with e.g. `-dev4108`
    # and pandas pre-releases which end with e.g. .dev0+618.gb552dc95c9
    version_str = version if isinstance(version, str) else version.__version__
    version_str = re.sub(r"(\D?dev.*$)", "", version_str)
    return tuple(int(re.sub(r"\D", "", v)) for v in version_str.split("."))


@overload
def isinstance_or_issubclass(
    obj_or_cls: type, cls_or_tuple: type[_T]
) -> TypeIs[type[_T]]: ...


@overload
def isinstance_or_issubclass(
    obj_or_cls: object | type, cls_or_tuple: type[_T]
) -> TypeIs[_T | type[_T]]: ...


@overload
def isinstance_or_issubclass(
    obj_or_cls: type, cls_or_tuple: tuple[type[_T1], type[_T2]]
) -> TypeIs[type[_T1 | _T2]]: ...


@overload
def isinstance_or_issubclass(
    obj_or_cls: object | type, cls_or_tuple: tuple[type[_T1], type[_T2]]
) -> TypeIs[_T1 | _T2 | type[_T1 | _T2]]: ...


@overload
def isinstance_or_issubclass(
    obj_or_cls: type, cls_or_tuple: tuple[type[_T1], type[_T2], type[_T3]]
) -> TypeIs[type[_T1 | _T2 | _T3]]: ...


@overload
def isinstance_or_issubclass(
    obj_or_cls: object | type, cls_or_tuple: tuple[type[_T1], type[_T2], type[_T3]]
) -> TypeIs[_T1 | _T2 | _T3 | type[_T1 | _T2 | _T3]]: ...


@overload
def isinstance_or_issubclass(
    obj_or_cls: Any, cls_or_tuple: tuple[type, ...]
) -> TypeIs[Any]: ...


def isinstance_or_issubclass(obj_or_cls: Any, cls_or_tuple: Any) -> bool:
    from narwhals.dtypes import DType

    if isinstance(obj_or_cls, DType):
        return isinstance(obj_or_cls, cls_or_tuple)
    return isinstance(obj_or_cls, cls_or_tuple) or (
        isinstance(obj_or_cls, type) and issubclass(obj_or_cls, cls_or_tuple)
    )


def validate_laziness(items: Iterable[Any]) -> None:
    from narwhals.dataframe import DataFrame
    from narwhals.dataframe import LazyFrame

    if all(isinstance(item, DataFrame) for item in items) or (
        all(isinstance(item, LazyFrame) for item in items)
    ):
        return
    msg = f"The items to concatenate should either all be eager, or all lazy, got: {[type(item) for item in items]}"
    raise TypeError(msg)


def maybe_align_index(
    lhs: FrameOrSeriesT, rhs: Series[Any] | DataFrame[Any] | LazyFrame[Any]
) -> FrameOrSeriesT:
    """Align `lhs` to the Index of `rhs`, if they're both pandas-like.

    Arguments:
        lhs: Dataframe or Series.
        rhs: Dataframe or Series to align with.

    Returns:
        Same type as input.

    Notes:
        This is only really intended for backwards-compatibility purposes,
        for example if your library already aligns indices for users.
        If you're designing a new library, we highly encourage you to not
        rely on the Index.
        For non-pandas-like inputs, this only checks that `lhs` and `rhs`
        are the same length.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals as nw
        >>> df_pd = pd.DataFrame({"a": [1, 2]}, index=[3, 4])
        >>> s_pd = pd.Series([6, 7], index=[4, 3])
        >>> df = nw.from_native(df_pd)
        >>> s = nw.from_native(s_pd, series_only=True)
        >>> nw.to_native(nw.maybe_align_index(df, s))
           a
        4  2
        3  1
    """
    from narwhals._pandas_like.dataframe import PandasLikeDataFrame
    from narwhals._pandas_like.series import PandasLikeSeries

    def _validate_index(index: Any) -> None:
        if not index.is_unique:
            msg = "given index doesn't have a unique index"
            raise ValueError(msg)

    lhs_any = cast("Any", lhs)
    rhs_any = cast("Any", rhs)
    if isinstance(
        getattr(lhs_any, "_compliant_frame", None), PandasLikeDataFrame
    ) and isinstance(getattr(rhs_any, "_compliant_frame", None), PandasLikeDataFrame):
        _validate_index(lhs_any._compliant_frame.native.index)
        _validate_index(rhs_any._compliant_frame.native.index)
        return lhs_any._with_compliant(
            lhs_any._compliant_frame._with_native(
                lhs_any._compliant_frame.native.loc[rhs_any._compliant_frame.native.index]
            )
        )
    if isinstance(
        getattr(lhs_any, "_compliant_frame", None), PandasLikeDataFrame
    ) and isinstance(getattr(rhs_any, "_compliant_series", None), PandasLikeSeries):
        _validate_index(lhs_any._compliant_frame.native.index)
        _validate_index(rhs_any._compliant_series.native.index)
        return lhs_any._with_compliant(
            lhs_any._compliant_frame._with_native(
                lhs_any._compliant_frame.native.loc[
                    rhs_any._compliant_series.native.index
                ]
            )
        )
    if isinstance(
        getattr(lhs_any, "_compliant_series", None), PandasLikeSeries
    ) and isinstance(getattr(rhs_any, "_compliant_frame", None), PandasLikeDataFrame):
        _validate_index(lhs_any._compliant_series.native.index)
        _validate_index(rhs_any._compliant_frame.native.index)
        return lhs_any._with_compliant(
            lhs_any._compliant_series._with_native(
                lhs_any._compliant_series.native.loc[
                    rhs_any._compliant_frame.native.index
                ]
            )
        )
    if isinstance(
        getattr(lhs_any, "_compliant_series", None), PandasLikeSeries
    ) and isinstance(getattr(rhs_any, "_compliant_series", None), PandasLikeSeries):
        _validate_index(lhs_any._compliant_series.native.index)
        _validate_index(rhs_any._compliant_series.native.index)
        return lhs_any._with_compliant(
            lhs_any._compliant_series._with_native(
                lhs_any._compliant_series.native.loc[
                    rhs_any._compliant_series.native.index
                ]
            )
        )
    if len(lhs_any) != len(rhs_any):
        msg = f"Expected `lhs` and `rhs` to have the same length, got {len(lhs_any)} and {len(rhs_any)}"
        raise ValueError(msg)
    return lhs


def maybe_get_index(obj: DataFrame[Any] | LazyFrame[Any] | Series[Any]) -> Any | None:
    """Get the index of a DataFrame or a Series, if it's pandas-like.

    Arguments:
        obj: Dataframe or Series.

    Returns:
        Same type as input.

    Notes:
        This is only really intended for backwards-compatibility purposes,
        for example if your library already aligns indices for users.
        If you're designing a new library, we highly encourage you to not
        rely on the Index.
        For non-pandas-like inputs, this returns `None`.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals as nw
        >>> df_pd = pd.DataFrame({"a": [1, 2], "b": [4, 5]})
        >>> df = nw.from_native(df_pd)
        >>> nw.maybe_get_index(df)
        RangeIndex(start=0, stop=2, step=1)
        >>> series_pd = pd.Series([1, 2])
        >>> series = nw.from_native(series_pd, series_only=True)
        >>> nw.maybe_get_index(series)
        RangeIndex(start=0, stop=2, step=1)
    """
    obj_any = cast("Any", obj)
    native_obj = obj_any.to_native()
    if is_pandas_like_dataframe(native_obj) or is_pandas_like_series(native_obj):
        return native_obj.index
    return None


def maybe_set_index(
    obj: FrameOrSeriesT,
    column_names: str | list[str] | None = None,
    *,
    index: Series[IntoSeriesT] | list[Series[IntoSeriesT]] | None = None,
) -> FrameOrSeriesT:
    """Set the index of a DataFrame or a Series, if it's pandas-like.

    Arguments:
        obj: object for which maybe set the index (can be either a Narwhals `DataFrame`
            or `Series`).
        column_names: name or list of names of the columns to set as index.
            For dataframes, only one of `column_names` and `index` can be specified but
            not both. If `column_names` is passed and `df` is a Series, then a
            `ValueError` is raised.
        index: series or list of series to set as index.

    Returns:
        Same type as input.

    Raises:
        ValueError: If one of the following condition happens:

            - none of `column_names` and `index` are provided
            - both `column_names` and `index` are provided
            - `column_names` is provided and `df` is a Series

    Notes:
        This is only really intended for backwards-compatibility purposes, for example if
        your library already aligns indices for users.
        If you're designing a new library, we highly encourage you to not
        rely on the Index.

        For non-pandas-like inputs, this is a no-op.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals as nw
        >>> df_pd = pd.DataFrame({"a": [1, 2], "b": [4, 5]})
        >>> df = nw.from_native(df_pd)
        >>> nw.to_native(nw.maybe_set_index(df, "b"))  # doctest: +NORMALIZE_WHITESPACE
           a
        b
        4  1
        5  2
    """
    from narwhals.translate import to_native

    df_any = cast("Any", obj)
    native_obj = df_any.to_native()

    if column_names is not None and index is not None:
        msg = "Only one of `column_names` or `index` should be provided"
        raise ValueError(msg)

    if not column_names and index is None:
        msg = "Either `column_names` or `index` should be provided"
        raise ValueError(msg)

    if index is not None:
        keys = (
            [to_native(idx, pass_through=True) for idx in index]
            if _is_iterable(index)
            else to_native(index, pass_through=True)
        )
    else:
        keys = column_names

    if is_pandas_like_dataframe(native_obj):
        return df_any._with_compliant(
            df_any._compliant_frame._with_native(native_obj.set_index(keys))
        )
    elif is_pandas_like_series(native_obj):
        from narwhals._pandas_like.utils import set_index

        if column_names:
            msg = "Cannot set index using column names on a Series"
            raise ValueError(msg)

        native_obj = set_index(
            native_obj,
            keys,
            implementation=obj._compliant_series._implementation,  # type: ignore[union-attr]
            backend_version=obj._compliant_series._backend_version,  # type: ignore[union-attr]
        )
        return df_any._with_compliant(df_any._compliant_series._with_native(native_obj))
    else:
        return df_any


def maybe_reset_index(obj: FrameOrSeriesT) -> FrameOrSeriesT:
    """Reset the index to the default integer index of a DataFrame or a Series, if it's pandas-like.

    Arguments:
        obj: Dataframe or Series.

    Returns:
        Same type as input.

    Notes:
        This is only really intended for backwards-compatibility purposes,
        for example if your library already resets the index for users.
        If you're designing a new library, we highly encourage you to not
        rely on the Index.
        For non-pandas-like inputs, this is a no-op.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals as nw
        >>> df_pd = pd.DataFrame({"a": [1, 2], "b": [4, 5]}, index=([6, 7]))
        >>> df = nw.from_native(df_pd)
        >>> nw.to_native(nw.maybe_reset_index(df))
           a  b
        0  1  4
        1  2  5
        >>> series_pd = pd.Series([1, 2])
        >>> series = nw.from_native(series_pd, series_only=True)
        >>> nw.maybe_get_index(series)
        RangeIndex(start=0, stop=2, step=1)
    """
    obj_any = cast("Any", obj)
    native_obj = obj_any.to_native()
    if is_pandas_like_dataframe(native_obj):
        native_namespace = obj_any.__native_namespace__()
        if _has_default_index(native_obj, native_namespace):
            return obj_any
        return obj_any._with_compliant(
            obj_any._compliant_frame._with_native(native_obj.reset_index(drop=True))
        )
    if is_pandas_like_series(native_obj):
        native_namespace = obj_any.__native_namespace__()
        if _has_default_index(native_obj, native_namespace):
            return obj_any
        return obj_any._with_compliant(
            obj_any._compliant_series._with_native(native_obj.reset_index(drop=True))
        )
    return obj_any


def _is_range_index(obj: Any, native_namespace: Any) -> TypeIs[pd.RangeIndex]:
    return isinstance(obj, native_namespace.RangeIndex)


# NOTE: Remove ignore(s) after release w/ (https://github.com/pandas-dev/pandas-stubs/pull/1115)
def _has_default_index(
    native_frame_or_series: pd.Series[Any] | pd.DataFrame, native_namespace: Any
) -> bool:
    index = native_frame_or_series.index
    return (
        _is_range_index(index, native_namespace)
        and index.start == 0
        and index.stop == len(index)
        and index.step == 1
    )


def maybe_convert_dtypes(
    obj: FrameOrSeriesT, *args: bool, **kwargs: bool | str
) -> FrameOrSeriesT:
    """Convert columns or series to the best possible dtypes using dtypes supporting ``pd.NA``, if df is pandas-like.

    Arguments:
        obj: DataFrame or Series.
        *args: Additional arguments which gets passed through.
        **kwargs: Additional arguments which gets passed through.

    Returns:
        Same type as input.

    Notes:
        For non-pandas-like inputs, this is a no-op.
        Also, `args` and `kwargs` just get passed down to the underlying library as-is.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals as nw
        >>> import numpy as np
        >>> df_pd = pd.DataFrame(
        ...     {
        ...         "a": pd.Series([1, 2, 3], dtype=np.dtype("int32")),
        ...         "b": pd.Series([True, False, np.nan], dtype=np.dtype("O")),
        ...     }
        ... )
        >>> df = nw.from_native(df_pd)
        >>> nw.to_native(
        ...     nw.maybe_convert_dtypes(df)
        ... ).dtypes  # doctest: +NORMALIZE_WHITESPACE
        a             Int32
        b           boolean
        dtype: object
    """
    obj_any = cast("Any", obj)
    native_obj = obj_any.to_native()
    if is_pandas_like_dataframe(native_obj):
        return obj_any._with_compliant(
            obj_any._compliant_frame._with_native(
                native_obj.convert_dtypes(*args, **kwargs)
            )
        )
    if is_pandas_like_series(native_obj):
        return obj_any._with_compliant(
            obj_any._compliant_series._with_native(
                native_obj.convert_dtypes(*args, **kwargs)
            )
        )
    return obj_any


def scale_bytes(sz: int, unit: SizeUnit) -> int | float:
    """Scale size in bytes to other size units (eg: "kb", "mb", "gb", "tb").

    Arguments:
        sz: original size in bytes
        unit: size unit to convert into

    Returns:
        Integer or float.
    """
    if unit in {"b", "bytes"}:
        return sz
    elif unit in {"kb", "kilobytes"}:
        return sz / 1024
    elif unit in {"mb", "megabytes"}:
        return sz / 1024**2
    elif unit in {"gb", "gigabytes"}:
        return sz / 1024**3
    elif unit in {"tb", "terabytes"}:
        return sz / 1024**4
    else:
        msg = f"`unit` must be one of {{'b', 'kb', 'mb', 'gb', 'tb'}}, got {unit!r}"
        raise ValueError(msg)


def is_ordered_categorical(series: Series[Any]) -> bool:
    """Return whether indices of categories are semantically meaningful.

    This is a convenience function to accessing what would otherwise be
    the `is_ordered` property from the DataFrame Interchange Protocol,
    see https://data-apis.org/dataframe-protocol/latest/API.html.

    - For Polars:
      - Enums are always ordered.
      - Categoricals are ordered if `dtype.ordering == "physical"`.
    - For pandas-like APIs:
      - Categoricals are ordered if `dtype.cat.ordered == True`.
    - For PyArrow table:
      - Categoricals are ordered if `dtype.type.ordered == True`.

    Arguments:
        series: Input Series.

    Returns:
        Whether the Series is an ordered categorical.

    Examples:
        >>> import narwhals as nw
        >>> import pandas as pd
        >>> import polars as pl
        >>> data = ["x", "y"]
        >>> s_pd = pd.Series(data, dtype=pd.CategoricalDtype(ordered=True))
        >>> s_pl = pl.Series(data, dtype=pl.Categorical(ordering="physical"))

        Let's define a library-agnostic function:

        >>> @nw.narwhalify
        ... def func(s):
        ...     return nw.is_ordered_categorical(s)

        Then, we can pass any supported library to `func`:

        >>> func(s_pd)
        True
        >>> func(s_pl)
        True
    """
    from narwhals._interchange.series import InterchangeSeries

    dtypes = series._compliant_series._version.dtypes
    compliant = series._compliant_series
    if isinstance(compliant, InterchangeSeries) and isinstance(
        series.dtype, dtypes.Categorical
    ):
        return compliant.native.describe_categorical["is_ordered"]
    if series.dtype == dtypes.Enum:
        return True
    if series.dtype != dtypes.Categorical:
        return False
    native_series = series.to_native()
    if is_polars_series(native_series):
        return native_series.dtype.ordering == "physical"  # type: ignore[attr-defined]
    if is_pandas_series(native_series):
        return bool(native_series.cat.ordered)
    if is_modin_series(native_series):  # pragma: no cover
        return native_series.cat.ordered
    if is_cudf_series(native_series):  # pragma: no cover
        return native_series.cat.ordered
    if is_pyarrow_chunked_array(native_series):
        from narwhals._arrow.utils import is_dictionary

        return is_dictionary(native_series.type) and native_series.type.ordered
    # If it doesn't match any of the above, let's just play it safe and return False.
    return False  # pragma: no cover


def generate_unique_token(
    n_bytes: int, columns: Sequence[str]
) -> str:  # pragma: no cover
    msg = (
        "Use `generate_temporary_column_name` instead. `generate_unique_token` is "
        "deprecated and it will be removed in future versions"
    )
    issue_deprecation_warning(msg, _version="1.13.0")
    return generate_temporary_column_name(n_bytes=n_bytes, columns=columns)


def generate_temporary_column_name(n_bytes: int, columns: Sequence[str]) -> str:
    """Generates a unique column name that is not present in the given list of columns.

    It relies on [python secrets token_hex](https://docs.python.org/3/library/secrets.html#secrets.token_hex)
    function to return a string nbytes random bytes.

    Arguments:
        n_bytes: The number of bytes to generate for the token.
        columns: The list of columns to check for uniqueness.

    Returns:
        A unique token that is not present in the given list of columns.

    Raises:
        AssertionError: If a unique token cannot be generated after 100 attempts.

    Examples:
        >>> import narwhals as nw
        >>> columns = ["abc", "xyz"]
        >>> nw.generate_temporary_column_name(n_bytes=8, columns=columns) not in columns
        True
    """
    counter = 0
    while True:
        token = token_hex(n_bytes)
        if token not in columns:
            return token

        counter += 1
        if counter > 100:
            msg = (
                "Internal Error: Narwhals was not able to generate a column name with "
                f"{n_bytes=} and not in {columns}"
            )
            raise AssertionError(msg)


def parse_columns_to_drop(
    compliant_frame: Any,
    columns: Iterable[str],
    strict: bool,  # noqa: FBT001
) -> list[str]:
    cols = compliant_frame.columns
    to_drop = list(columns)
    if strict:
        missing_columns = [x for x in to_drop if x not in cols]
        if missing_columns:
            raise ColumnNotFoundError.from_missing_and_available_column_names(
                missing_columns=missing_columns, available_columns=cols
            )
    else:
        to_drop = list(set(cols).intersection(set(to_drop)))
    return to_drop


def is_sequence_but_not_str(sequence: Sequence[_T] | Any) -> TypeIs[Sequence[_T]]:
    return isinstance(sequence, Sequence) and not isinstance(sequence, str)


def is_slice_none(obj: Any) -> TypeIs[_SliceNone]:
    return isinstance(obj, slice) and obj == slice(None)


def is_sized_multi_index_selector(
    obj: Any,
) -> TypeIs[SizedMultiIndexSelector[Series[Any] | CompliantSeries[Any]]]:
    return (
        (
            is_sequence_but_not_str(obj)
            and ((len(obj) > 0 and isinstance(obj[0], int)) or (len(obj) == 0))
        )
        or is_numpy_array_1d_int(obj)
        or is_narwhals_series_int(obj)
        or is_compliant_series_int(obj)
    )


def is_sequence_like(
    obj: Sequence[_T] | Any,
) -> TypeIs[Sequence[_T] | Series[Any] | _1DArray]:
    return (
        is_sequence_but_not_str(obj)
        or is_numpy_array_1d(obj)
        or is_narwhals_series(obj)
        or is_compliant_series(obj)
    )


def is_slice_index(obj: Any) -> TypeIs[_SliceIndex]:
    return isinstance(obj, slice) and (
        isinstance(obj.start, int)
        or isinstance(obj.stop, int)
        or (isinstance(obj.step, int) and obj.start is None and obj.stop is None)
    )


def is_range(obj: Any) -> TypeIs[range]:
    return isinstance(obj, range)


def is_single_index_selector(obj: Any) -> TypeIs[SingleIndexSelector]:
    return bool(isinstance(obj, int) and not isinstance(obj, bool))


def is_index_selector(
    obj: Any,
) -> TypeIs[SingleIndexSelector | MultiIndexSelector[Series[Any] | CompliantSeries[Any]]]:
    return (
        is_single_index_selector(obj)
        or is_sized_multi_index_selector(obj)
        or is_slice_index(obj)
    )


def is_list_of(obj: Any, tp: type[_T]) -> TypeIs[list[_T]]:
    # Check if an object is a list of `tp`, only sniffing the first element.
    return bool(isinstance(obj, list) and obj and isinstance(obj[0], tp))


def is_sequence_of(obj: Any, tp: type[_T]) -> TypeIs[Sequence[_T]]:
    # Check if an object is a sequence of `tp`, only sniffing the first element.
    return bool(
        is_sequence_but_not_str(obj)
        and (first := next(iter(obj), None))
        and isinstance(first, tp)
    )


def find_stacklevel() -> int:
    """Find the first place in the stack that is not inside narwhals.

    Returns:
        Stacklevel.

    Taken from:
    https://github.com/pandas-dev/pandas/blob/ab89c53f48df67709a533b6a95ce3d911871a0a8/pandas/util/_exceptions.py#L30-L51
    """
    import inspect
    from pathlib import Path

    import narwhals as nw

    pkg_dir = str(Path(nw.__file__).parent)

    # https://stackoverflow.com/questions/17407119/python-inspect-stack-is-slow
    frame = inspect.currentframe()
    n = 0
    try:
        while frame:
            fname = inspect.getfile(frame)
            if fname.startswith(pkg_dir) or (
                (qualname := getattr(frame.f_code, "co_qualname", None))
                # ignore @singledispatch wrappers
                and qualname.startswith("singledispatch.")
            ):
                frame = frame.f_back
                n += 1
            else:  # pragma: no cover
                break
        else:  # pragma: no cover
            pass
    finally:
        # https://docs.python.org/3/library/inspect.html
        # > Though the cycle detector will catch these, destruction of the frames
        # > (and local variables) can be made deterministic by removing the cycle
        # > in a finally clause.
        del frame
    return n


def issue_deprecation_warning(message: str, _version: str) -> None:
    """Issue a deprecation warning.

    Arguments:
        message: The message associated with the warning.
        _version: Narwhals version when the warning was introduced. Just used for internal
            bookkeeping.
    """
    warn(message=message, category=DeprecationWarning, stacklevel=find_stacklevel())


def validate_strict_and_pass_though(
    strict: bool | None,  # noqa: FBT001
    pass_through: bool | None,  # noqa: FBT001
    *,
    pass_through_default: bool,
    emit_deprecation_warning: bool,
) -> bool:
    if strict is None and pass_through is None:
        pass_through = pass_through_default
    elif strict is not None and pass_through is None:
        if emit_deprecation_warning:
            msg = (
                "`strict` in `from_native` is deprecated, please use `pass_through` instead.\n\n"
                "Note: `strict` will remain available in `narwhals.stable.v1`.\n"
                "See https://narwhals-dev.github.io/narwhals/backcompat/ for more information.\n"
            )
            issue_deprecation_warning(msg, _version="1.13.0")
        pass_through = not strict
    elif strict is None and pass_through is not None:
        pass
    else:
        msg = "Cannot pass both `strict` and `pass_through`"
        raise ValueError(msg)
    return pass_through


def deprecate_native_namespace(
    *, warn_version: str = "", required: bool = False
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to transition from `native_namespace` to `backend` argument.

    Arguments:
        warn_version: Emit a deprecation warning from this version.
        required: Raise when both `native_namespace`, `backend` are `None`.

    Returns:
        Wrapped function, with `native_namespace` **removed**.
    """

    def decorate(fn: Callable[P, R], /) -> Callable[P, R]:
        @wraps(fn)
        def wrapper(*args: P.args, **kwds: P.kwargs) -> R:
            backend = kwds.pop("backend", None)
            native_namespace = kwds.pop("native_namespace", None)
            if native_namespace is not None and backend is None:
                if warn_version:
                    msg = (
                        "`native_namespace` is deprecated, please use `backend` instead.\n\n"
                        "Note: `native_namespace` will remain available in `narwhals.stable.v1`.\n"
                        "See https://narwhals-dev.github.io/narwhals/backcompat/ for more information.\n"
                    )
                    issue_deprecation_warning(msg, _version=warn_version)
                backend = native_namespace
            elif native_namespace is not None and backend is not None:
                msg = "Can't pass both `native_namespace` and `backend`"
                raise ValueError(msg)
            elif native_namespace is None and backend is None and required:
                msg = f"`backend` must be specified in `{fn.__name__}`."
                raise ValueError(msg)
            kwds["backend"] = backend
            return fn(*args, **kwds)

        return wrapper

    return decorate


def _validate_rolling_arguments(
    window_size: int, min_samples: int | None
) -> tuple[int, int]:
    if window_size < 1:
        msg = "window_size must be greater or equal than 1"
        raise ValueError(msg)

    if not isinstance(window_size, int):
        _type = window_size.__class__.__name__
        msg = (
            f"argument 'window_size': '{_type}' object cannot be "
            "interpreted as an integer"
        )
        raise TypeError(msg)

    if min_samples is not None:
        if min_samples < 1:
            msg = "min_samples must be greater or equal than 1"
            raise ValueError(msg)

        if not isinstance(min_samples, int):
            _type = min_samples.__class__.__name__
            msg = (
                f"argument 'min_samples': '{_type}' object cannot be "
                "interpreted as an integer"
            )
            raise TypeError(msg)
        if min_samples > window_size:
            msg = "`min_samples` must be less or equal than `window_size`"
            raise InvalidOperationError(msg)
    else:
        min_samples = window_size

    return window_size, min_samples


def generate_repr(header: str, native_repr: str) -> str:
    try:
        terminal_width = os.get_terminal_size().columns
    except OSError:
        terminal_width = int(os.getenv("COLUMNS", 80))  # noqa: PLW1508
    native_lines = native_repr.expandtabs().splitlines()
    max_native_width = max(len(line) for line in native_lines)

    if max_native_width + 2 <= terminal_width:
        length = max(max_native_width, len(header))
        output = f"┌{'─' * length}┐\n"
        header_extra = length - len(header)
        output += f"|{' ' * (header_extra // 2)}{header}{' ' * (header_extra // 2 + header_extra % 2)}|\n"
        output += f"|{'-' * (length)}|\n"
        start_extra = (length - max_native_width) // 2
        end_extra = (length - max_native_width) // 2 + (length - max_native_width) % 2
        for line in native_lines:
            output += f"|{' ' * (start_extra)}{line}{' ' * (end_extra + max_native_width - len(line))}|\n"
        output += f"└{'─' * length}┘"
        return output

    diff = 39 - len(header)
    return (
        f"┌{'─' * (39)}┐\n"
        f"|{' ' * (diff // 2)}{header}{' ' * (diff // 2 + diff % 2)}|\n"
        "| Use `.to_native` to see native output |\n└"
        f"{'─' * 39}┘"
    )


def check_column_exists(columns: Sequence[str], subset: Sequence[str] | None) -> None:
    if subset is not None and (missing := set(subset).difference(columns)):
        msg = f"Column(s) {sorted(missing)} not found in {columns}"
        raise ColumnNotFoundError(msg)


def check_column_names_are_unique(columns: Sequence[str]) -> None:
    len_unique_columns = len(set(columns))
    if len(columns) != len_unique_columns:
        from collections import Counter

        counter = Counter(columns)
        duplicates = {k: v for k, v in counter.items() if v > 1}
        msg = "".join(f"\n- '{k}' {v} times" for k, v in duplicates.items())
        msg = f"Expected unique column names, got:{msg}"
        raise DuplicateError(msg)


def _parse_time_unit_and_time_zone(
    time_unit: TimeUnit | Iterable[TimeUnit] | None,
    time_zone: str | timezone | Iterable[str | timezone | None] | None,
) -> tuple[Set[TimeUnit], Set[str | None]]:
    time_units: Set[TimeUnit] = (
        {"ms", "us", "ns", "s"}
        if time_unit is None
        else {time_unit}
        if isinstance(time_unit, str)
        else set(time_unit)
    )
    time_zones: Set[str | None] = (
        {None}
        if time_zone is None
        else {str(time_zone)}
        if isinstance(time_zone, (str, timezone))
        else {str(tz) if tz is not None else None for tz in time_zone}
    )
    return time_units, time_zones


def dtype_matches_time_unit_and_time_zone(
    dtype: DType, dtypes: DTypes, time_units: Set[TimeUnit], time_zones: Set[str | None]
) -> bool:
    return (
        isinstance(dtype, dtypes.Datetime)
        and (dtype.time_unit in time_units)
        and (
            dtype.time_zone in time_zones
            or ("*" in time_zones and dtype.time_zone is not None)
        )
    )


def get_column_names(frame: _StoresColumns, /) -> Sequence[str]:
    return frame.columns


def exclude_column_names(frame: _StoresColumns, names: Container[str]) -> Sequence[str]:
    return [col_name for col_name in frame.columns if col_name not in names]


def passthrough_column_names(names: Sequence[str], /) -> EvalNames[Any]:
    def fn(_frame: Any, /) -> Sequence[str]:
        return names

    return fn


def _hasattr_static(obj: Any, attr: str) -> bool:
    sentinel = object()
    return getattr_static(obj, attr, sentinel) is not sentinel


def is_compliant_dataframe(
    obj: CompliantDataFrame[CompliantSeriesT, CompliantExprT, NativeFrameT_co] | Any,
) -> TypeIs[CompliantDataFrame[CompliantSeriesT, CompliantExprT, NativeFrameT_co]]:
    return _hasattr_static(obj, "__narwhals_dataframe__")


def is_compliant_lazyframe(
    obj: CompliantLazyFrame[CompliantExprT, NativeFrameT_co] | Any,
) -> TypeIs[CompliantLazyFrame[CompliantExprT, NativeFrameT_co]]:
    return _hasattr_static(obj, "__narwhals_lazyframe__")


def is_compliant_series(
    obj: CompliantSeries[NativeSeriesT_co] | Any,
) -> TypeIs[CompliantSeries[NativeSeriesT_co]]:
    return _hasattr_static(obj, "__narwhals_series__")


def is_compliant_series_int(
    obj: CompliantSeries[NativeSeriesT_co] | Any,
) -> TypeIs[CompliantSeries[NativeSeriesT_co]]:
    return is_compliant_series(obj) and obj.dtype.is_integer()


def is_compliant_expr(
    obj: CompliantExpr[CompliantFrameT, CompliantSeriesOrNativeExprT_co] | Any,
) -> TypeIs[CompliantExpr[CompliantFrameT, CompliantSeriesOrNativeExprT_co]]:
    return hasattr(obj, "__narwhals_expr__")


def is_eager_allowed(obj: Implementation) -> TypeIs[EagerAllowedImplementation]:
    return obj in {
        Implementation.PANDAS,
        Implementation.MODIN,
        Implementation.CUDF,
        Implementation.POLARS,
        Implementation.PYARROW,
    }


def has_native_namespace(obj: Any) -> TypeIs[SupportsNativeNamespace]:
    return hasattr(obj, "__native_namespace__")


def _supports_dataframe_interchange(obj: Any) -> TypeIs[DataFrameLike]:
    return hasattr(obj, "__dataframe__")


def supports_arrow_c_stream(obj: Any) -> TypeIs[ArrowStreamExportable]:
    return _hasattr_static(obj, "__arrow_c_stream__")


def _remap_full_join_keys(
    left_on: Sequence[str], right_on: Sequence[str], suffix: str
) -> dict[str, str]:
    """Remap join keys to avoid collisions.

    If left keys collide with the right keys, append the suffix.
    If there's no collision, let the right keys be.

    Arguments:
        left_on: Left keys.
        right_on: Right keys.
        suffix: Suffix to append to right keys.

    Returns:
        A map of old to new right keys.
    """
    right_keys_suffixed = (
        f"{key}{suffix}" if key in left_on else key for key in right_on
    )
    return dict(zip(right_on, right_keys_suffixed))


def _into_arrow_table(data: IntoArrowTable, context: _FullContext, /) -> pa.Table:
    """Guards `ArrowDataFrame.from_arrow` w/ safer imports.

    Arguments:
        data: Object which implements `__arrow_c_stream__`.
        context: Initialized compliant object.

    Returns:
        A PyArrow Table.
    """
    if find_spec("pyarrow"):
        import pyarrow as pa  # ignore-banned-import

        from narwhals._arrow.namespace import ArrowNamespace

        version = context._version
        ns = ArrowNamespace(backend_version=parse_version(pa), version=version)
        return ns._dataframe.from_arrow(data, context=ns).native
    else:  # pragma: no cover
        msg = f"PyArrow>=14.0.0 is required for `from_arrow` for object of type {type(data).__name__!r}."
        raise ModuleNotFoundError(msg)


# TODO @dangotbanned: Extend with runtime behavior for `v1.*`
# See `narwhals.exceptions.NarwhalsUnstableWarning`
def unstable(fn: _Fn, /) -> _Fn:
    """Visual-only marker for unstable functionality.

    Arguments:
        fn: Function to decorate.

    Returns:
        Decorated function (unchanged).

    Examples:
        >>> from narwhals.utils import unstable
        >>> @unstable
        ... def a_work_in_progress_feature(*args):
        ...     return args
        >>>
        >>> a_work_in_progress_feature.__name__
        'a_work_in_progress_feature'
        >>> a_work_in_progress_feature(1, 2, 3)
        (1, 2, 3)
    """
    return fn


if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 13):
        # NOTE: avoids `mypy`
        #     error: Module "narwhals.utils" does not explicitly export attribute "deprecated"  [attr-defined]
        from warnings import deprecated as deprecated  # noqa: PLC0414
    else:
        from typing_extensions import deprecated as deprecated  # noqa: PLC0414
else:

    def deprecated(message: str, /) -> Callable[[_Fn], _Fn]:  # noqa: ARG001
        def wrapper(func: _Fn, /) -> _Fn:
            return func

        return wrapper


class not_implemented:  # noqa: N801
    """Mark some functionality as unsupported.

    Arguments:
        alias: optional name used instead of the data model hook [`__set_name__`].

    Returns:
        An exception-raising [descriptor].

    Notes:
        - Attribute/method name *doesn't* need to be declared twice
        - Allows different behavior when looked up on the class vs instance
        - Allows us to use `isinstance(...)` instead of monkeypatching an attribute to the function

    Examples:
        >>> from narwhals.utils import not_implemented
        >>> class Thing:
        ...     def totally_ready(self) -> str:
        ...         return "I'm ready!"
        ...
        ...     not_ready_yet = not_implemented()
        >>>
        >>> thing = Thing()
        >>> thing.totally_ready()
        "I'm ready!"
        >>> thing.not_ready_yet()
        Traceback (most recent call last):
            ...
        NotImplementedError: 'not_ready_yet' is not implemented for: 'Thing'.
        ...
        >>> isinstance(Thing.not_ready_yet, not_implemented)
        True

    [`__set_name__`]: https://docs.python.org/3/reference/datamodel.html#object.__set_name__
    [descriptor]: https://docs.python.org/3/howto/descriptor.html
    """

    def __init__(self, alias: str | None = None, /) -> None:
        # NOTE: Don't like this
        # Trying to workaround `mypy` requiring `@property` everywhere
        self._alias: str | None = alias

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>: {self._name_owner}.{self._name}"

    def __set_name__(self, owner: type[_T], name: str) -> None:
        # https://docs.python.org/3/howto/descriptor.html#customized-names
        self._name_owner: str = owner.__name__
        self._name: str = self._alias or name

    def __get__(
        self, instance: _T | Literal["raise"] | None, owner: type[_T] | None = None, /
    ) -> Any:
        if instance is None:
            # NOTE: Branch for `cls._name`
            # We can check that to see if an instance of `type(self)` for
            # https://narwhals-dev.github.io/narwhals/api-completeness/expr/
            return self
        # NOTE: Prefer not exposing the actual class we're defining in
        # `_implementation` may not be available everywhere
        who = getattr(instance, "_implementation", self._name_owner)
        raise _not_implemented_error(self._name, who)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # NOTE: Purely to duck-type as assignable to **any** instance method
        # Wouldn't be reachable through *regular* attribute access
        return self.__get__("raise")

    @classmethod
    def deprecated(cls, message: LiteralString, /) -> Self:
        """Alt constructor, wraps with `@deprecated`.

        Arguments:
            message: **Static-only** deprecation message, emitted in an IDE.

        Returns:
            An exception-raising [descriptor].

        [descriptor]: https://docs.python.org/3/howto/descriptor.html
        """
        obj = cls()
        return deprecated(message)(obj)


def _not_implemented_error(what: str, who: str, /) -> NotImplementedError:
    msg = (
        f"{what!r} is not implemented for: {who!r}.\n\n"
        "If you would like to see this functionality in `narwhals`, "
        "please open an issue at: https://github.com/narwhals-dev/narwhals/issues"
    )
    return NotImplementedError(msg)


class requires:  # noqa: N801
    """Method decorator for raising under certain constraints.

    Attributes:
        _min_version: Minimum backend version.
        _hint: Optional suggested alternative.

    Examples:
        >>> from narwhals.utils import requires, Implementation
        >>> class SomeBackend:
        ...     _implementation = Implementation.PYARROW
        ...     _backend_version = 20, 0, 0
        ...
        ...     @requires.backend_version((9000, 0, 0))
        ...     def really_complex_feature(self) -> str:
        ...         return "hello"
        >>> backend = SomeBackend()
        >>> backend.really_complex_feature()
        Traceback (most recent call last):
            ...
        NotImplementedError: `really_complex_feature` is only available in PyArrow>='9000.0.0', found version '20.0.0'.
    """

    _min_version: tuple[int, ...]
    _hint: str

    @classmethod
    def backend_version(cls, minimum: tuple[int, ...], /, hint: str = "") -> Self:
        """Method decorator for raising below a minimum `_backend_version`.

        Arguments:
            minimum: Minimum backend version.
            hint: Optional suggested alternative.

        Returns:
            An exception-raising decorator.
        """
        obj = cls.__new__(cls)
        obj._min_version = minimum
        obj._hint = hint
        return obj

    @staticmethod
    def _unparse_version(backend_version: tuple[int, ...], /) -> str:
        return ".".join(f"{d}" for d in backend_version)

    def _ensure_version(self, instance: _FullContext, /) -> None:
        if instance._backend_version >= self._min_version:
            return
        method = self._wrapped_name
        backend = instance._implementation._alias
        minimum = self._unparse_version(self._min_version)
        found = self._unparse_version(instance._backend_version)
        msg = f"`{method}` is only available in {backend}>={minimum!r}, found version {found!r}."
        if self._hint:
            msg = f"{msg}\n{self._hint}"
        raise NotImplementedError(msg)

    def __call__(self, fn: _Method[_ContextT, P, R], /) -> _Method[_ContextT, P, R]:
        self._wrapped_name = fn.__name__

        @wraps(fn)
        def wrapper(instance: _ContextT, *args: P.args, **kwds: P.kwargs) -> R:
            self._ensure_version(instance)
            return fn(instance, *args, **kwds)

        # NOTE: Only getting a complaint from `mypy`
        return wrapper  # type: ignore[return-value]


def convert_str_slice_to_int_slice(
    str_slice: _SliceName, columns: Sequence[str]
) -> tuple[int | None, int | None, Any]:
    start = columns.index(str_slice.start) if str_slice.start is not None else None
    stop = columns.index(str_slice.stop) + 1 if str_slice.stop is not None else None
    step = str_slice.step
    return (start, stop, step)


def inherit_doc(
    tp_parent: Callable[P, R1], /
) -> Callable[[_Constructor[_T, P, R2]], _Constructor[_T, P, R2]]:
    """Steal the class-level docstring from parent and attach to child `__init__`.

    Returns:
        Decorated constructor.

    Notes:
        - Passes static typing (mostly)
        - Passes at runtime
    """

    def decorate(init_child: _Constructor[_T, P, R2], /) -> _Constructor[_T, P, R2]:
        if init_child.__name__ == "__init__" and issubclass(type(tp_parent), type):
            init_child.__doc__ = getdoc(tp_parent)
            return init_child
        else:  # pragma: no cover
            msg = (
                f"`@{inherit_doc.__name__}` is only allowed to decorate an `__init__` with a class-level doc.\n"
                f"Method: {init_child.__qualname__!r}\n"
                f"Parent: {tp_parent!r}"
            )
            raise TypeError(msg)

    return decorate
