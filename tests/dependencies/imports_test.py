from __future__ import annotations

from importlib.util import find_spec
from types import ModuleType

import pytest

from narwhals._utils import backend_version
from narwhals.utils import Implementation

implementations = [
    Implementation.CUDF,
    Implementation.DASK,
    Implementation.DUCKDB,
    Implementation.IBIS,
    Implementation.MODIN,
    Implementation.PANDAS,
    Implementation.POLARS,
    Implementation.PYARROW,
    Implementation.PYSPARK,
    Implementation.SQLFRAME,
]


@pytest.mark.parametrize("impl", implementations)
def test_to_native_namespace(impl: Implementation) -> None:
    if not find_spec(impl.value):
        reason = f"{impl.value} not installed"
        pytest.skip(reason=reason)

    assert isinstance(impl.to_native_namespace(), ModuleType)


@pytest.mark.parametrize("impl", implementations)
def test_to_native_namespace_min_version(
    monkeypatch: pytest.MonkeyPatch, impl: Implementation
) -> None:
    if not find_spec(impl.value):
        reason = f"{impl.value} not installed"
        pytest.skip(reason=reason)

    monkeypatch.setattr("narwhals._utils.parse_version", lambda _: (0, 0, 1))
    backend_version.cache_clear()

    with pytest.raises(ValueError, match="Minimum version"):
        impl.to_native_namespace()


def test_to_native_namespace_unknown() -> None:
    impl = Implementation.UNKNOWN
    with pytest.raises(
        AssertionError, match="Cannot return native namespace from UNKNOWN Implementation"
    ):
        impl.to_native_namespace()
