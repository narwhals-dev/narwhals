from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from narwhals.dependencies import is_into_dataframe
from narwhals.stable.v1.dependencies import is_into_dataframe as v1_is_into_dataframe

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Self

    from tests.utils import Constructor, ConstructorEager

EAGER_CONSTRUCTOR_NAMES = ("pandas", "modin", "cudf", "polars_eager", "pyarrow")
V1_INTO_DATAFRAMES = (*EAGER_CONSTRUCTOR_NAMES, "duckdb", "ibis")

data: dict[str, Any] = {"a": [1, 2, 3], "b": [4, 5, 6]}


class DictDataFrame:
    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = data

    def __len__(self) -> int:  # pragma: no cover
        return len(next(iter(self._data.values())))

    def __narwhals_dataframe__(self) -> Self:  # pragma: no cover
        return self


def test_is_into_dataframe_lazy(constructor: Constructor) -> None:
    if any(x in str(constructor) for x in EAGER_CONSTRUCTOR_NAMES):
        assert is_into_dataframe(constructor(data))
    else:
        assert not is_into_dataframe(constructor(data))

    if any(x in str(constructor) for x in V1_INTO_DATAFRAMES):
        assert v1_is_into_dataframe(constructor(data))
    else:
        assert not v1_is_into_dataframe(constructor(data))


def test_is_into_dataframe_eager(constructor_eager: ConstructorEager) -> None:
    assert is_into_dataframe(constructor_eager(data))
    assert v1_is_into_dataframe(constructor_eager(data))


def test_is_into_dataframe_other() -> None:
    pytest.importorskip("numpy")
    import numpy as np

    assert is_into_dataframe(DictDataFrame(data))  # pyrefly: ignore[bad-specialization]
    assert not is_into_dataframe(np.array([[1, 4], [2, 5], [3, 6]]))
    assert not is_into_dataframe(data)

    assert v1_is_into_dataframe(DictDataFrame(data))  # pyrefly: ignore[bad-specialization]
    assert not v1_is_into_dataframe(np.array([[1, 4], [2, 5], [3, 6]]))
    assert not v1_is_into_dataframe(data)
