from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Sequence

import pytest

import narwhals as nw
from narwhals._namespace import Namespace
from narwhals.utils import Version

if TYPE_CHECKING:
    from typing_extensions import TypeAlias
    from typing_extensions import assert_type

    from narwhals._arrow.namespace import ArrowNamespace  # noqa: F401
    from narwhals._namespace import BackendName
    from narwhals._namespace import _EagerAllowed
    from narwhals._pandas_like.namespace import PandasLikeNamespace  # noqa: F401
    from narwhals._polars.namespace import PolarsNamespace  # noqa: F401
    from tests.utils import Constructor

IntoIterable: TypeAlias = Callable[[Sequence[Any]], Iterable[Any]]

_EAGER_ALLOWED = "polars", "pandas", "pyarrow", "modin", "cudf"
_LAZY_ONLY = "dask", "duckdb", "pyspark", "sqlframe"
_LAZY_ALLOWED = ("polars", *_LAZY_ONLY)
_BACKENDS = (*_EAGER_ALLOWED, *_LAZY_ONLY)

eager_allowed = pytest.mark.parametrize("backend", _EAGER_ALLOWED)
lazy_allowed = pytest.mark.parametrize("backend", _LAZY_ALLOWED)
backends = pytest.mark.parametrize("backend", _BACKENDS)


@backends
def test_namespace_from_backend_name(backend: BackendName) -> None:
    pytest.importorskip(backend)
    namespace = Namespace.from_backend(backend)
    assert namespace.implementation.name.lower() == backend
    assert namespace.version is Version.MAIN


def test_namespace_from_native_object(constructor: Constructor) -> None:
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    frame = constructor(data)
    # NOTE: Needed to remove those from `IntoFrame`
    # See (#2239)
    assert not isinstance(frame, (nw.DataFrame, nw.LazyFrame))

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
    # BUG: `@overload`(s) for `from_native` fail to mentioned `Compliant` support
    # Fix *should be*:
    #     `IntoSeries: TypeAlias = Series[Any] | NativeSeries | CompliantSeries[Any]`
    # Can't do that as, `PolarsSeries` is not compliant yet
    series = nw.from_native(compliant_series, series_only=True)  # pyright: ignore[reportCallIssue, reportArgumentType]
    assert isinstance(series, nw.Series)
    assert series.to_list() == list(data)
