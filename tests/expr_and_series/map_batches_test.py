from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("numpy")

import narwhals as nw
from tests.utils import (
    PANDAS_VERSION,
    POLARS_VERSION,
    ConstructorEager,
    assert_equal_data,
)

if TYPE_CHECKING:
    from narwhals.dtypes import DType

data = {"a": [1, 2, 3], "b": [4, 5, 6], "z": [7.0, 8.0, 9.0]}


def test_map_batches_expr_compliant(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    expected = df.select(nw.col("a", "b").map_batches(lambda s: s + 1).name.suffix("1"))
    assert_equal_data(expected, {"a1": [2, 3, 4], "b1": [5, 6, 7]})


@pytest.mark.parametrize(
    ("value", "dtype"),
    [(1, nw.Int64()), ("foo", nw.String()), ([1, 2], nw.List(nw.Int64()))],
)
def test_map_batches_expr_scalar(
    constructor_eager: ConstructorEager, value: Any, dtype: DType
) -> None:
    df = nw.from_native(constructor_eager(data))
    if dtype.is_nested() and df.implementation.is_pandas_like():
        if PANDAS_VERSION < (2, 2):  # pragma: no cover
            reason = "pandas is too old for nested types"
            pytest.skip(reason=reason)
        pytest.importorskip("pyarrow")

    expected = df.select(
        nw.col("a", "b").map_batches(
            lambda _: value, returns_scalar=True, return_dtype=dtype
        )
    )
    assert_equal_data(expected, {"a": [value], "b": [value]})


def test_map_batches_expr_numpy_array(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    expected = df.select(
        nw.col("a")
        .map_batches(lambda s: s.to_numpy() + 1, return_dtype=nw.Float64())
        .sum()
    )
    assert_equal_data(expected, {"a": [9.0]})


def test_map_batches_expr_numpy_scalar(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))

    expected = df.select(
        nw.all().map_batches(lambda s: s.to_numpy().argmax(), returns_scalar=True)
    )
    assert_equal_data(expected, {"a": [2], "b": [2], "z": [2]})


def test_map_batches_expr_names(constructor_eager: ConstructorEager) -> None:
    df = nw.from_native(constructor_eager(data))
    expected = nw.from_native(df.select(nw.all().map_batches(lambda x: x.to_numpy())))
    assert_equal_data(expected, {"a": [1, 2, 3], "b": [4, 5, 6], "z": [7.0, 8.0, 9.0]})


def test_map_batches_exception(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "polars" in str(constructor_eager) and POLARS_VERSION < (1, 32, 3):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor_eager(data))
    msg = (
        r"`map(?:_batches)?` with `returns_scalar=False` must return a Series; found "
        "'numpy.int64'.\n\nIf `returns_scalar` is set to `True`, a returned value can be "
        "a scalar value."
    )

    with pytest.raises(TypeError, match=msg):
        df.select(nw.all().map_batches(lambda s: s.to_numpy().argmax()))
