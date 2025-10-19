from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals._utils import Implementation
from tests.utils import Constructor, assert_equal_data

if TYPE_CHECKING:
    from narwhals._typing import EagerAllowed, Polars


def test_from_dict(eager_backend: EagerAllowed) -> None:
    result = nw.DataFrame.from_dict({"c": [1, 2], "d": [5, 6]}, backend=eager_backend)
    expected = {"c": [1, 2], "d": [5, 6]}
    assert_equal_data(result, expected)
    assert isinstance(result, nw.DataFrame)


def test_from_dict_schema(eager_backend: EagerAllowed) -> None:
    schema = {"c": nw.Int16(), "d": nw.Float32()}
    result = nw.DataFrame.from_dict(
        {"c": [1, 2], "d": [5, 6]}, backend=eager_backend, schema=schema
    )
    assert result.collect_schema() == schema


@pytest.mark.parametrize("backend", [Implementation.POLARS, "polars"])
def test_from_dict_without_backend(constructor: Constructor, backend: Polars) -> None:
    pytest.importorskip("polars")
    pytest.importorskip("pyarrow")

    df = (
        nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
        .lazy()
        .collect(backend=backend)
    )
    result = nw.DataFrame.from_dict({"c": df["a"], "d": df["b"]})
    assert_equal_data(result, {"c": [1, 2, 3], "d": [4, 5, 6]})


def test_from_dict_without_backend_invalid(constructor: Constructor) -> None:
    df = nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]})).lazy().collect()
    with pytest.raises(TypeError, match="backend"):
        nw.DataFrame.from_dict({"c": nw.to_native(df["a"]), "d": nw.to_native(df["b"])})


def test_from_dict_with_backend_invalid() -> None:
    pytest.importorskip("duckdb")
    with pytest.raises(ValueError, match="lazy-only"):
        nw.DataFrame.from_dict({"c": [1, 2], "d": [5, 6]}, backend="duckdb")  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", [Implementation.POLARS, "polars"])
def test_from_dict_one_native_one_narwhals(
    constructor: Constructor, backend: Polars
) -> None:
    pytest.importorskip("pyarrow")
    pytest.importorskip("polars")

    df = (
        nw.from_native(constructor({"a": [1, 2, 3], "b": [4, 5, 6]}))
        .lazy()
        .collect(backend=backend)
    )
    result = nw.DataFrame.from_dict({"c": nw.to_native(df["a"]), "d": df["b"]})
    expected = {"c": [1, 2, 3], "d": [4, 5, 6]}
    assert_equal_data(result, expected)


def test_from_dict_empty(eager_backend: EagerAllowed) -> None:
    result = nw.DataFrame.from_dict({}, backend=eager_backend)
    assert result.shape == (0, 0)


def test_from_dict_empty_with_schema(eager_backend: EagerAllowed) -> None:
    schema = nw.Schema({"a": nw.String(), "b": nw.Int8()})
    result = nw.DataFrame.from_dict({}, schema, backend=eager_backend)
    assert result.schema == schema


def test_alignment() -> None:
    pytest.importorskip("pandas")
    import pandas as pd

    # https://github.com/narwhals-dev/narwhals/issues/1474
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = nw.DataFrame.from_dict(
        {"a": df["a"], "b": df["a"].sort_values(ascending=False)}, backend=pd
    ).to_native()
    expected = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})
    pd.testing.assert_frame_equal(result, expected)


def test_from_dict_object_2851(
    eager_backend: EagerAllowed, request: pytest.FixtureRequest
) -> None:
    data = {"Var1": [3, "a"], "Var2": ["a", "b"]}
    schema = {"Var1": nw.Object(), "Var2": nw.String()}
    request.applymarker(
        pytest.mark.xfail(
            "pyarrow" in str(eager_backend),
            reason="Object DType not supported in pyarrow",
            raises=NotImplementedError,
        )
    )
    df = nw.DataFrame.from_dict(data, backend=eager_backend, schema=schema)
    assert df.schema == schema
