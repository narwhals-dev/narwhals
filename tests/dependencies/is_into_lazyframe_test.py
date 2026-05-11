from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from narwhals.stable.v1.dependencies import is_into_lazyframe

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from typing_extensions import Self

    from tests.utils import Constructor, ConstructorEager

EAGER_CONSTRUCTOR_NAMES = ("pandas", "modin", "cudf", "polars_eager", "pyarrow")

data: dict[str, Any] = {"a": [1, 2, 3], "b": [4, 5, 6]}


class LazyDictDataFrame:
    def __init__(self, data: Mapping[str, Sequence[Any]]) -> None:
        self._data = data

    def __len__(self) -> int:  # pragma: no cover
        return len(next(iter(self._data.values())))

    def __narwhals_lazyframe__(self) -> Self:  # pragma: no cover
        return self


def test_is_into_lazyframe_lazy(constructor: Constructor) -> None:
    if any(x in str(constructor) for x in EAGER_CONSTRUCTOR_NAMES):
        assert not is_into_lazyframe(constructor(data))
    else:
        assert is_into_lazyframe(constructor(data))


def test_is_into_lazyframe_eager(constructor_eager: ConstructorEager) -> None:
    assert not is_into_lazyframe(constructor_eager(data))


def test_is_into_lazyframe_other() -> None:
    pytest.importorskip("numpy")
    import numpy as np

    assert is_into_lazyframe(LazyDictDataFrame(data))  # pyrefly: ignore[bad-specialization]
    assert not is_into_lazyframe(np.array([[1, 4], [2, 5], [3, 6]]))
    assert not is_into_lazyframe(data)
