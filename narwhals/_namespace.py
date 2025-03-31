"""Narwhals-level equivalent of `CompliantNamespace`.

Aiming to solve 2 distinct issues.

### 1. A unified entry point for creating a `CompliantNamespace`

Currently lots of ways we do this:
- Most recently `nw.utils._into_compliant_namespace`
- Creating an object, then using `__narwhals_namespace__`
- Generally repeating logic in multiple places


### 2. Typing and no `lambda`s for `nw.(expr|functions)`

Lacking a better alternative, the current pattern is:

    lambda plx: plx.all()
    lambda plx: apply_n_ary_operation(
        plx, lambda x, y: x - y, self, other, str_as_lit=True
    )

If this can *also* get those parts typed - then 🎉
"""

# ruff: noqa: PYI042
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import Literal
from typing import overload

from narwhals._compliant.typing import CompliantNamespaceAny
from narwhals._compliant.typing import CompliantNamespaceT_co
from narwhals.utils import Implementation
from narwhals.utils import Version

if TYPE_CHECKING:
    from types import ModuleType
    from typing import ClassVar

    from typing_extensions import TypeAlias

    from narwhals._arrow.namespace import ArrowNamespace
    from narwhals._dask.namespace import DaskNamespace
    from narwhals._duckdb.namespace import DuckDBNamespace
    from narwhals._pandas_like.namespace import PandasLikeNamespace
    from narwhals._polars.namespace import PolarsNamespace
    from narwhals._spark_like.namespace import SparkLikeNamespace
    from narwhals.utils import _Arrow
    from narwhals.utils import _Dask
    from narwhals.utils import _DuckDB
    from narwhals.utils import _EagerAllowed
    from narwhals.utils import _EagerOnly
    from narwhals.utils import _LazyAllowed
    from narwhals.utils import _LazyOnly
    from narwhals.utils import _PandasLike
    from narwhals.utils import _Polars
    from narwhals.utils import _SparkLike

    _polars: TypeAlias = Literal["polars"]
    _arrow: TypeAlias = Literal["pyarrow"]
    _dask: TypeAlias = Literal["dask"]
    _duckdb: TypeAlias = Literal["duckdb"]
    _pandas_like: TypeAlias = Literal["pandas", "cudf", "modin"]
    _spark_like: TypeAlias = Literal["pyspark", "sqlframe"]
    _eager_only: TypeAlias = "_pandas_like | _arrow"
    _eager_allowed: TypeAlias = "_polars | _eager_only"
    _lazy_only: TypeAlias = "_spark_like | _dask | _duckdb"
    _lazy_allowed: TypeAlias = "_polars | _lazy_only"
    BackendName: TypeAlias = "_eager_allowed | _lazy_allowed"

    Polars: TypeAlias = "_polars | _Polars"
    Arrow: TypeAlias = "_arrow | _Arrow"
    Dask: TypeAlias = "_dask | _Dask"
    DuckDB: TypeAlias = "_duckdb | _DuckDB"
    PandasLike: TypeAlias = "_pandas_like | _PandasLike"
    SparkLike: TypeAlias = "_spark_like | _SparkLike"
    EagerOnly: TypeAlias = "_eager_only | _EagerOnly"
    EagerAllowed: TypeAlias = "_eager_allowed | _EagerAllowed"
    LazyOnly: TypeAlias = "_lazy_only | _LazyOnly"
    LazyAllowed: TypeAlias = "_lazy_allowed | _LazyAllowed"

    IntoBackend: TypeAlias = "BackendName | Implementation | ModuleType"

__all__ = ["Namespace"]


class Namespace(Generic[CompliantNamespaceT_co]):
    _compliant_namespace: CompliantNamespaceT_co
    _version: ClassVar[Version] = Version.MAIN

    def __init__(self, namespace: CompliantNamespaceT_co, /) -> None:
        self._compliant_namespace = namespace

    def __repr__(self) -> str:
        return f"Namespace[{type(self.compliant).__name__}]"

    @property
    def compliant(self) -> CompliantNamespaceT_co:
        return self._compliant_namespace

    @property
    def implementation(self) -> Implementation:
        return self.compliant._implementation

    @property
    def version(self) -> Version:
        return self._version

    @overload
    @classmethod
    def from_backend(cls, backend: PandasLike, /) -> Namespace[PandasLikeNamespace]: ...

    @overload
    @classmethod
    def from_backend(cls, backend: Polars, /) -> Namespace[PolarsNamespace]: ...

    @overload
    @classmethod
    def from_backend(cls, backend: Arrow, /) -> Namespace[ArrowNamespace]: ...

    @overload
    @classmethod
    def from_backend(cls, backend: SparkLike, /) -> Namespace[SparkLikeNamespace]: ...

    @overload
    @classmethod
    def from_backend(cls, backend: DuckDB, /) -> Namespace[DuckDBNamespace]: ...

    @overload
    @classmethod
    def from_backend(cls, backend: Dask, /) -> Namespace[DaskNamespace]: ...

    @overload
    @classmethod
    def from_backend(cls, backend: ModuleType, /) -> Namespace[CompliantNamespaceAny]: ...

    @classmethod
    def from_backend(
        cls: type[Namespace[Any]], backend: IntoBackend, /
    ) -> Namespace[Any]:
        """Instantiate from native namespace module, string, or Implementation.

        Arguments:
            backend: native namespace module, string, or Implementation.

        Returns:
            Namespace.

        Examples:
            >>> Namespace.from_backend("polars")
            'Namespace[PolarsNamespace]'
        """
        impl = Implementation.from_backend(backend)
        backend_version = impl._backend_version
        version = cls._version
        ns: CompliantNamespaceAny
        if impl.is_pandas_like():
            from narwhals._pandas_like.namespace import PandasLikeNamespace

            ns = PandasLikeNamespace(
                implementation=impl, backend_version=backend_version, version=version
            )

        elif impl.is_polars():
            from narwhals._polars.namespace import PolarsNamespace

            ns = PolarsNamespace(backend_version=backend_version, version=version)
        elif impl.is_pyarrow():
            from narwhals._arrow.namespace import ArrowNamespace

            ns = ArrowNamespace(backend_version=backend_version, version=version)
        elif impl.is_spark_like():  # pragma: no cover
            from narwhals._spark_like.namespace import SparkLikeNamespace

            ns = SparkLikeNamespace(
                implementation=impl, backend_version=backend_version, version=version
            )
        elif impl.is_duckdb():  # pragma: no cover
            from narwhals._duckdb.namespace import DuckDBNamespace

            ns = DuckDBNamespace(backend_version=backend_version, version=version)
        elif impl.is_dask():  # pragma: no cover
            from narwhals._dask.namespace import DaskNamespace

            ns = DaskNamespace(backend_version=backend_version, version=version)
        else:
            msg = "Not supported Implementation"  # pragma: no cover
            raise AssertionError(msg)
        return cls(ns)
