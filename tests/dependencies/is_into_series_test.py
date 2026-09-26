from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
import narwhals.stable.v2 as nw_v2
from narwhals.dependencies import is_into_series
from narwhals.stable.v1.dependencies import is_into_series as v1_is_into_series
from narwhals.stable.v2.dependencies import is_into_series as v2_is_into_series

if TYPE_CHECKING:
    from typing_extensions import Self

    from tests.dependencies.conftest import AlwaysHasAttr
    from tests.utils import ConstructorEager

data: dict[str, Any] = {"a": [1, 2, 3], "b": [4, 5, 6]}


class ListBackedSeries:  # pragma: no cover
    def __init__(self, name: str, data: list[Any]) -> None:
        self._data = data
        self._name = name

    def __narwhals_series__(self) -> Self:
        return self

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Any: ...
    def filter(self, *args: Any, **kwargs: Any) -> Any: ...
    def value_counts(self, *args: Any, **kwargs: Any) -> Any: ...
    def unique(self, *args: Any, **kwargs: Any) -> Any: ...


def test_is_into_series(constructor_eager: ConstructorEager) -> None:
    native_frame = constructor_eager(data).to_native()
    nw_series = nw.from_native(native_frame)["a"]
    nw_v1_series = nw_v1.from_native(native_frame)["a"]
    nw_v2_series = nw_v2.from_native(native_frame)["a"]
    native_series = nw_series.to_native()

    assert is_into_series(native_series)
    assert is_into_series(nw_series)
    assert is_into_series(nw_v1_series)
    assert is_into_series(nw_v2_series)

    assert v1_is_into_series(native_series)
    assert not v1_is_into_series(nw_series)
    assert v1_is_into_series(nw_v1_series)
    assert not v1_is_into_series(nw_v2_series)

    assert v2_is_into_series(native_series)
    assert not v2_is_into_series(nw_series)
    assert not v2_is_into_series(nw_v1_series)
    assert v2_is_into_series(nw_v2_series)


def test_is_into_series_numpy() -> None:
    pytest.importorskip("numpy")
    import numpy as np

    arr = np.array([1, 2, 3])
    assert not is_into_series(arr)
    assert not v1_is_into_series(arr)
    assert not v2_is_into_series(arr)


def test_is_into_series_other(always_has_attr: AlwaysHasAttr) -> None:
    values = [1, 4, 2]

    assert not is_into_series(values)
    assert not v1_is_into_series(values)
    assert not v2_is_into_series(values)

    assert is_into_series(ListBackedSeries("a", values))
    assert v1_is_into_series(ListBackedSeries("a", values))
    assert v2_is_into_series(ListBackedSeries("a", values))

    assert not is_into_series(always_has_attr)
    assert not v1_is_into_series(always_has_attr)
    assert not v2_is_into_series(always_has_attr)
