from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals._namespace import Namespace

if TYPE_CHECKING:
    from narwhals._namespace import BackendName
    from tests.utils import Constructor


@pytest.mark.parametrize(
    "name",
    [
        "polars",
        "pandas",
        "pyarrow",
        "dask",
        "duckdb",
        "pyspark",
        "sqlframe",
        "modin",
        "cudf",
    ],
)
def test_namespace_from_backend_name(name: BackendName) -> None:
    pytest.importorskip(name)
    namespace = Namespace.from_backend(name)
    assert namespace.implementation.name.lower() == name


def test_namespace_from_native_object(constructor: Constructor) -> None:
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    frame = constructor(data)
    namespace = Namespace.from_native_object(frame)
    nw_frame = nw.from_native(frame)
    assert namespace.implementation == nw_frame.implementation
