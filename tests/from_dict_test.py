from __future__ import annotations

from datetime import datetime
from importlib.util import find_spec

import pandas as pd
import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from narwhals.utils import Implementation
from tests.utils import Constructor
from tests.utils import assert_equal_data

TEST_EAGER_BACKENDS: list[Implementation | str] = []
TEST_EAGER_BACKENDS.extend(
    (Implementation.POLARS, "polars") if find_spec("polars") is not None else ()
)
TEST_EAGER_BACKENDS.extend(
    (Implementation.PANDAS, "pandas") if find_spec("pandas") is not None else ()
)
TEST_EAGER_BACKENDS.extend(
    (Implementation.PYARROW, "pyarrow") if find_spec("pyarrow") is not None else ()
)


@pytest.mark.parametrize(
    "backend",
    TEST_EAGER_BACKENDS,
)
def test_from_dict(
    backend: Implementation | str,
) -> None:
    result = nw.from_dict({"c": [1, 2], "d": [5, 6]}, backend=backend)
    expected = {"c": [1, 2], "d": [5, 6]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw.DataFrame)


@pytest.mark.parametrize(
    "backend",
    TEST_EAGER_BACKENDS,
)
def test_from_dict_schema(
    backend: Implementation | str,
) -> None:
    schema = {"c": nw_v1.Int16(), "d": nw_v1.Float32()}
    result = nw_v1.from_dict({"c": [1, 2], "d": [5, 6]}, backend=backend, schema=schema)
    assert result.collect_schema() == schema


@pytest.mark.parametrize(
    "backend",
    [
        Implementation.POLARS,
        "polars",
    ],
)
def test_from_dict_without_backend(
    constructor: Constructor, backend: Implementation | str
) -> None:
    pytest.importorskip("polars")

    df = (
        nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
        .lazy()
        .collect(backend=backend)
    )
    result = nw.from_dict({"c": df["a"], "d": df["b"]})
    assert_equal_data(result, {"c": [1, 2, 3], "d": [4, 5, 6]})


def test_from_dict_without_backend_invalid(
    constructor: Constructor,
) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]})).lazy().collect()
    with pytest.raises(TypeError, match="backend"):
        nw.from_dict({"c": nw.to_native(df["a"]), "d": nw.to_native(df["b"])})


def test_from_dict_with_backend_invalid() -> None:
    pytest.importorskip("duckdb")
    with pytest.raises(ValueError, match="Unsupported `backend` value"):
        nw.from_dict({"c": [1, 2], "d": [5, 6]}, backend="duckdb")


def test_from_dict_both_backend_and_namespace(
    constructor: Constructor,
) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
    native_namespace = nw.get_native_namespace(df)
    with pytest.raises(ValueError, match="Can't pass both"):
        nw.from_dict(
            {"c": [1, 2], "d": [5, 6]},
            backend="pandas",
            native_namespace=native_namespace,
        )


def test_from_dict_both_backend_and_namespace_v1(
    constructor: Constructor,
) -> None:
    df = nw_v1.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
    native_namespace = nw_v1.get_native_namespace(df)
    with pytest.raises(ValueError, match="Can't pass both"):
        nw_v1.from_dict(
            {"c": [1, 2], "d": [5, 6]},
            backend="pandas",
            native_namespace=native_namespace,
        )


@pytest.mark.parametrize(
    "backend",
    [
        Implementation.POLARS,
        "polars",
    ],
)
def test_from_dict_one_native_one_narwhals(
    constructor: Constructor, backend: Implementation | str
) -> None:
    pytest.importorskip("polars")

    df = (
        nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
        .lazy()
        .collect(backend=backend)
    )
    result = nw.from_dict({"c": nw.to_native(df["a"]), "d": df["b"]})
    expected = {"c": [1, 2, 3], "d": [4, 5, 6]}
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    "backend",
    TEST_EAGER_BACKENDS,
)
def test_from_dict_v1(
    backend: Implementation | str,
) -> None:
    result = nw_v1.from_dict(
        {"c": [1, 2], "d": [datetime(2020, 1, 1), datetime(2020, 1, 2)]},
        backend=backend,
    )
    expected = {"c": [1, 2], "d": [datetime(2020, 1, 1), datetime(2020, 1, 2)]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw_v1.DataFrame)
    assert isinstance(result.schema["d"], nw_v1.dtypes.Datetime)


def test_from_dict_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        nw.from_dict({})


def test_alignment() -> None:
    # https://github.com/narwhals-dev/narwhals/issues/1474
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = nw.from_dict(
        {"a": df["a"], "b": df["a"].sort_values(ascending=False)}, backend=pd
    ).to_native()
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    pd.testing.assert_frame_equal(result, expected)
