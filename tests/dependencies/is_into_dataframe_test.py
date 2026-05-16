from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
import narwhals.stable.v2 as nw_v2
from narwhals.dependencies import is_into_dataframe
from narwhals.stable.v1.dependencies import is_into_dataframe as v1_is_into_dataframe
from narwhals.stable.v2.dependencies import is_into_dataframe as v2_is_into_dataframe

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Self

    from tests.dependencies.conftest import DynamicAttrOnly
    from tests.utils import Constructor

EAGER_CONSTRUCTOR_NAMES = ("pandas", "modin", "cudf", "polars_eager", "pyarrow")
V1_INTO_DATAFRAMES = (*EAGER_CONSTRUCTOR_NAMES, "duckdb", "ibis")

data: dict[str, Any] = {"a": [1, 2, 3], "b": [4, 5, 6]}


class DictDataFrame:  # pragma: no cover
    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = data

    def __narwhals_dataframe__(self) -> Self:
        return self

    def __len__(self) -> int:
        return len(next(iter(self._data.values())))

    @property
    def columns(self) -> Any: ...
    def drop(self, *args: Any, **kwargs: Any) -> Any: ...
    def join(self, *args: Any, **kwargs: Any) -> Any: ...


@pytest.mark.filterwarnings("ignore:.*You passed a.*:UserWarning")
def test_is_into_dataframe(constructor: Constructor) -> None:
    native_frame = constructor(data)
    nw_frame = nw.from_native(native_frame)
    nw_v1_frame = nw_v1.from_native(native_frame)
    nw_v2_frame = nw_v2.from_native(native_frame)

    result = any(x in str(constructor) for x in EAGER_CONSTRUCTOR_NAMES)
    assert is_into_dataframe(native_frame) == result
    assert is_into_dataframe(nw_frame) == result

    result_v1 = any(x in str(constructor) for x in V1_INTO_DATAFRAMES)
    assert v1_is_into_dataframe(native_frame) == result_v1
    assert v1_is_into_dataframe(nw_v1_frame) == result_v1
    assert v1_is_into_dataframe(nw_v2_frame) is False

    result_v2 = any(x in str(constructor) for x in EAGER_CONSTRUCTOR_NAMES)
    assert v2_is_into_dataframe(native_frame) == result_v2
    assert v2_is_into_dataframe(nw_v2_frame) == result_v2
    assert v2_is_into_dataframe(nw_v1_frame) is False

    assert is_into_dataframe(nw_v1_frame) == result_v1
    assert is_into_dataframe(nw_v2_frame) == result_v2
    assert not v1_is_into_dataframe(nw_frame)
    assert not v2_is_into_dataframe(nw_frame)


def test_is_into_dataframe_numpy() -> None:
    pytest.importorskip("numpy")
    import numpy as np

    arr = np.array([[1, 4], [2, 5], [3, 6]])
    assert not is_into_dataframe(arr)
    assert not v1_is_into_dataframe(arr)
    assert not v2_is_into_dataframe(arr)


def test_is_into_dataframe_other(dynamic_attr_only: DynamicAttrOnly) -> None:
    assert not is_into_dataframe(data)
    assert not v1_is_into_dataframe(data)
    assert not v2_is_into_dataframe(data)

    assert is_into_dataframe(DictDataFrame(data))
    assert v1_is_into_dataframe(DictDataFrame(data))
    assert v2_is_into_dataframe(DictDataFrame(data))

    assert not is_into_dataframe(dynamic_attr_only)
    assert not v1_is_into_dataframe(dynamic_attr_only)
    assert not v2_is_into_dataframe(dynamic_attr_only)
