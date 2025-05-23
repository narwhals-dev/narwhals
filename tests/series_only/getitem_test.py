from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

import narwhals as nw
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


def test_by_slice(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [1, 4, 2]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = {"a": df["a"][[0, 1]]}
    expected = {"a": [1, 2]}
    assert_equal_data(result, expected)
    result = {"a": df["a"][1:]}
    expected = {"a": [2, 3]}
    assert_equal_data(result, expected)
    result = {"b": df[:, 1]}
    expected = {"b": [4, 5, 6]}
    assert_equal_data(result, expected)
    result = {"b": df[:, "b"]}
    expected = {"b": [4, 5, 6]}
    assert_equal_data(result, expected)
    result = {"b": df[:2, "b"]}
    expected = {"b": [4, 5]}
    assert_equal_data(result, expected)
    result = {"b": df[:2, 1]}
    expected = {"b": [4, 5]}
    assert_equal_data(result, expected)
    result = {"b": df[[0, 1], 1]}
    expected = {"b": [4, 5]}
    assert_equal_data(result, expected)
    result = {"b": df[[], 1]}
    expected = {"b": []}
    assert_equal_data(result, expected)


def test_getitem_arrow_scalar() -> None:
    result = nw.from_native(pa.chunked_array([[1]]), series_only=True)[0]
    assert isinstance(result, int)


def test_index(constructor_eager: ConstructorEager) -> None:
    df = constructor_eager({"a": [0, 1, 2]})
    snw = nw.from_native(df, eager_only=True)["a"]
    assert snw[snw[0]] == 0


@pytest.mark.filterwarnings(
    "ignore:.*_array__ implementation doesn't accept a copy keyword.*:DeprecationWarning:modin"
)
def test_getitem_other_series(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager({"a": [1, None, 2, 3]}), eager_only=True)[
        "a"
    ]
    other = nw.from_native(constructor_eager({"b": [1, 3]}), eager_only=True)["b"]
    assert_equal_data(series[other].to_frame(), {"a": [None, 3]})


def test_getitem_invalid_series(constructor_eager: ConstructorEager) -> None:
    series = nw.from_native(constructor_eager({"a": [1, None, 2, 3]}), eager_only=True)[
        "a"
    ]
    with pytest.raises(TypeError, match="Unexpected type"):
        series[series > 1]
