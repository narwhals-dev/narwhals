from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from narwhals.utils import Implementation

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


data = {"a": [1, 2, 3]}


def test_lazy(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.lazy()
    assert isinstance(result, nw.LazyFrame)
    df = nw_v1.from_native(constructor_eager(data), eager_only=True)
    result = df.lazy()
    assert isinstance(result, nw_v1.LazyFrame)


@pytest.mark.parametrize(
    "backend", [Implementation.POLARS, Implementation.DUCKDB, Implementation.DASK]
)
def test_lazy_backend(
    constructor_eager: ConstructorEager, backend: Implementation
) -> None:
    if backend is Implementation.DASK:
        pytest.importorskip("dask")
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.lazy(backend=backend)
    assert isinstance(result, nw.LazyFrame)
    assert result.implementation == backend
    df = nw_v1.from_native(constructor_eager(data), eager_only=True)
    result = df.lazy(backend=backend)
    assert isinstance(result, nw_v1.LazyFrame)
    assert result.implementation == backend


@pytest.mark.parametrize("backend", [Implementation.PANDAS])
def test_lazy_backend_invalid(
    constructor_eager: ConstructorEager, backend: Implementation
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    with pytest.raises(ValueError, match="Not supported backend"):
        df.lazy(backend=backend)
