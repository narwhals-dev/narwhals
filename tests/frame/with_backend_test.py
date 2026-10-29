from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals._utils import Implementation
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from narwhals._typing import EagerAllowed
    from tests.utils import ConstructorEager


data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}


@pytest.mark.slow
@pytest.mark.parametrize(
    "backend",
    [
        Implementation.PANDAS,
        Implementation.MODIN,
        Implementation.CUDF,
        Implementation.POLARS,
        Implementation.PYARROW,
        "pandas",
        "polars",
        "pyarrow",
        "modin",
        "cudf",
    ],
)
def test_with_backend(constructor_eager: ConstructorEager, backend: EagerAllowed) -> None:
    impl = Implementation.from_backend(backend)
    pytest.importorskip(impl.name.lower())

    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.with_backend(backend=backend)

    assert isinstance(result, nw.DataFrame)
    assert result.implementation == impl
    assert_equal_data(result, data)


@pytest.mark.parametrize(
    "backend",
    [
        Implementation.DUCKDB,
        Implementation.DASK,
        Implementation.IBIS,
        Implementation.PYSPARK,
        Implementation.SQLFRAME,
        "duckdb",
        "dask",
        "ibis",
        "pyspark",
        "sqlframe",
        "garbage",
    ],
)
def test_with_backend_invalid(
    constructor_eager: ConstructorEager, backend: str | Implementation
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    with pytest.raises(ValueError, match=re.escape("Unsupported `backend` value")):
        df.with_backend(backend=backend)  # type: ignore[arg-type]
