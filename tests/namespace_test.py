from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from narwhals._namespace import Namespace

if TYPE_CHECKING:
    from narwhals._namespace import BackendName


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
