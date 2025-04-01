from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals._namespace import Namespace

if TYPE_CHECKING:
    from narwhals._namespace import BackendName
    from tests.utils import Constructor

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


def test_namespace_from_native_object(constructor: Constructor) -> None:
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    frame = constructor(data)
    namespace = Namespace.from_native_object(frame)
    nw_frame = nw.from_native(frame)
    assert namespace.implementation == nw_frame.implementation
