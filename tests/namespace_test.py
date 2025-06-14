from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

import pytest

import narwhals as nw
from narwhals._namespace import Namespace
from narwhals._utils import Version

if TYPE_CHECKING:
    from typing_extensions import TypeAlias, assert_type

    from narwhals._arrow.namespace import ArrowNamespace  # noqa: F401
    from narwhals._compliant import CompliantNamespace
    from narwhals._compliant.typing import CompliantExprAny
    from narwhals._namespace import BackendName, _EagerAllowed
    from narwhals._pandas_like.expr import PandasLikeExpr
    from narwhals._pandas_like.namespace import PandasLikeNamespace  # noqa: F401
    from narwhals._polars.namespace import PolarsNamespace  # noqa: F401
    from narwhals.typing import _2DArray
    from tests.utils import Constructor

ExprT = TypeVar("ExprT", bound="CompliantExprAny")

IntoIterable: TypeAlias = Callable[[Sequence[Any]], Iterable[Any]]

_EAGER_ALLOWED = "polars", "pandas", "pyarrow", "modin", "cudf"
_LAZY_ONLY = "dask", "duckdb", "pyspark", "sqlframe"
_LAZY_ALLOWED = ("polars", *_LAZY_ONLY)
_BACKENDS = (*_EAGER_ALLOWED, *_LAZY_ONLY)

eager_allowed = pytest.mark.parametrize("backend", _EAGER_ALLOWED)
lazy_allowed = pytest.mark.parametrize("backend", _LAZY_ALLOWED)
backends = pytest.mark.parametrize("backend", _BACKENDS)


def _compliant_len(ns: CompliantNamespace[Any, ExprT]) -> ExprT:
    return ns.len()


@backends
def test_preserve_type_var(backend: BackendName) -> None:
    # If we have multiple *potential* backends and together they don't hit an `@overload`
    # we should fall back to `Any`.
    # However when we have a *single* backend, we should be able to yoink it's `TypeVar`s
    # out for use elsewhere.
    pytest.importorskip(backend)
    from_backend = Version.MAIN.namespace.from_backend
    namespace_any = from_backend(backend).compliant
    expr_any = _compliant_len(namespace_any)
    assert expr_any
    if TYPE_CHECKING:
        # NOTE: If this `Any` fails due to (future) improved inference in a type-checker,
        # the detail of this being `Any` can be swapped out for the new type
        assert_type(expr_any, Any)
        namespace_pandas = from_backend("pandas").compliant
        expr_pandas = _compliant_len(namespace_pandas)
        assert_type(expr_pandas, PandasLikeExpr)


@backends
def test_namespace_from_backend_name(backend: BackendName) -> None:
    pytest.importorskip(backend)
    namespace = Namespace.from_backend(backend)
    assert namespace.implementation.name.lower() == backend
    assert namespace.version is Version.MAIN


def test_namespace_from_native_object(constructor: Constructor) -> None:
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    frame = constructor(data)
    namespace = Namespace.from_native_object(frame)
    nw_frame = nw.from_native(frame)
    assert namespace.implementation == nw_frame.implementation


def test_namespace_from_native_object_invalid() -> None:
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    with pytest.raises(TypeError, match=r"dict"):
        Namespace.from_native_object(data)  # pyright: ignore[reportCallIssue, reportArgumentType]


@eager_allowed
def test_namespace_from_backend_typing(backend: _EagerAllowed) -> None:
    pytest.importorskip(backend)
    namespace = Namespace.from_backend(backend)
    if TYPE_CHECKING:
        assert_type(
            namespace,
            "Namespace[PolarsNamespace] | Namespace[PandasLikeNamespace] | Namespace[ArrowNamespace]",
        )
    assert repr(namespace) in {
        "Namespace[PolarsNamespace]",
        "Namespace[PandasLikeNamespace]",
        "Namespace[ArrowNamespace]",
    }


@pytest.mark.parametrize("into_iter", [list, tuple, deque, iter])
@eager_allowed
def test_namespace_series_from_iterable(
    backend: _EagerAllowed, into_iter: IntoIterable
) -> None:
    pytest.importorskip(backend)
    data = 1, 2, 3
    namespace = Namespace.from_backend(backend)
    compliant = namespace.compliant
    iterable = into_iter(data)
    compliant_series = compliant._series.from_iterable(
        iterable, context=compliant, name="hello"
    )
    series = nw.from_native(compliant_series, series_only=True)
    assert isinstance(series, nw.Series)
    assert series.to_list() == list(data)


def test_namespace_from_numpy_polars() -> None:
    pytest.importorskip("polars")
    pytest.importorskip("numpy")
    import numpy as np
    import polars as pl

    arr: _2DArray = cast("_2DArray", np.array([[5, 2, 0, 1], [1, 4, 7, 8], [1, 2, 3, 9]]))
    columns = "a", "b", "c"
    frame = Namespace.from_backend("polars").compliant.from_numpy(arr, columns).native
    if TYPE_CHECKING:
        assert_type(frame, pl.DataFrame)

    assert isinstance(frame, pl.DataFrame)
    assert frame.columns == list(columns)


def test_import_namespace() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("polars")
    import pandas as pd

    data = {"a": [1, 2, 3]}
    native = pd.DataFrame(data)
    ns = Version.V1.namespace.from_native_object(native)
    assert ns.version is Version.V1
    assert ns.implementation.is_pandas()

    ns = Version.MAIN.namespace.from_native_object(native)
    assert ns.version is Version.MAIN
    assert ns.implementation.is_pandas()

    compliant = ns.compliant._dataframe.from_dict(data, context=ns.compliant, schema=None)
    ns_pl = ns.compliant._version.namespace.from_native_object(compliant.to_polars())
    assert ns_pl.version is Version.MAIN
    assert ns_pl.implementation.is_polars()


def test_namespace_init_subclass() -> None:
    pytest.importorskip("polars")

    class NamespaceV1Any(Namespace[Any], version=Version.V1): ...

    ns_any = NamespaceV1Any.from_backend("polars")
    assert ns_any.version is Version.V1

    class NamespaceV1NoTypeVar(Namespace, version=Version.V1): ...  # type: ignore[type-arg]

    ns_no_type_var = NamespaceV1NoTypeVar.from_backend("polars")
    assert ns_no_type_var.version is Version.V1

    with pytest.raises(
        TypeError, match=re.compile(r"missing.+required.+argument.+version")
    ):

        class NamespaceNoVersion(Namespace): ...  # type: ignore[call-arg, type-arg]

    with pytest.raises(TypeError, match=re.compile(r"Expected.+Version.+but got.+str")):

        class NamespaceBadVersion(Namespace, version="invalid version"): ...  # type: ignore[arg-type, type-arg]
