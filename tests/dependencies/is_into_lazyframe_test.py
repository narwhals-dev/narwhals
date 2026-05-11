from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
from narwhals.dependencies import is_into_lazyframe
from narwhals.stable.v1.dependencies import is_into_lazyframe as v1_is_into_lazyframe

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Self

    from tests.utils import Constructor

EAGER_CONSTRUCTOR_NAMES = ("pandas", "modin", "cudf", "polars_eager", "pyarrow")
V1_INTO_DATAFRAMES = (*EAGER_CONSTRUCTOR_NAMES, "duckdb", "ibis")

data: dict[str, Any] = {"a": [1, 2, 3], "b": [4, 5, 6]}


class DictDataFrame:
    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = data

    def __len__(self) -> int:  # pragma: no cover
        return len(next(iter(self._data.values())))

    def __narwhals_lazyframe__(self) -> Self:  # pragma: no cover
        return self


@pytest.mark.filterwarnings("ignore:.*You passed a.*:UserWarning")
def test_is_into_lazyframe(constructor: Constructor) -> None:
    native_frame = constructor(data)
    nw_frame = nw.from_native(native_frame)
    nw_v1_frame = nw_v1.from_native(native_frame)

    result = not any(x in str(constructor) for x in EAGER_CONSTRUCTOR_NAMES)
    assert is_into_lazyframe(native_frame) == result
    assert is_into_lazyframe(nw_frame) == result

    result_v1 = not any(x in str(constructor) for x in V1_INTO_DATAFRAMES)
    assert v1_is_into_lazyframe(native_frame) == result_v1
    assert v1_is_into_lazyframe(nw_v1_frame) == result_v1

    assert is_into_lazyframe(nw_v1_frame) == result_v1
    assert not v1_is_into_lazyframe(nw_frame)


def test_is_into_lazyframe_other() -> None:
    pytest.importorskip("numpy")
    import numpy as np

    assert is_into_lazyframe(DictDataFrame(data))  # pyrefly: ignore[bad-specialization]
    assert not is_into_lazyframe(np.array([[1, 4], [2, 5], [3, 6]]))
    assert not is_into_lazyframe(data)

    assert v1_is_into_lazyframe(DictDataFrame(data))  # pyrefly: ignore[bad-specialization]
    assert not v1_is_into_lazyframe(np.array([[1, 4], [2, 5], [3, 6]]))
    assert not v1_is_into_lazyframe(data)
