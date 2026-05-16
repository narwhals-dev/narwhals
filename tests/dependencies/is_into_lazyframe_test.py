from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
import narwhals.stable.v2 as nw_v2
from narwhals.dependencies import is_into_lazyframe
from narwhals.stable.v1.dependencies import is_into_lazyframe as v1_is_into_lazyframe
from narwhals.stable.v2.dependencies import is_into_lazyframe as v2_is_into_lazyframe

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Self

    from tests.utils import Constructor

EAGER_CONSTRUCTOR_NAMES = ("pandas", "modin", "cudf", "polars_eager", "pyarrow")
V1_INTO_DATAFRAMES = (*EAGER_CONSTRUCTOR_NAMES, "duckdb", "ibis")

data: dict[str, Any] = {"a": [1, 2, 3], "b": [4, 5, 6]}


class DictLazyFrame:  # pragma: no cover
    def __init__(self, data: Mapping[str, Any]) -> None:
        self._data = data

    def __narwhals_lazyframe__(self) -> Self:
        return self

    @property
    def columns(self) -> Any: ...
    def drop(self, *args: Any, **kwargs: Any) -> Any: ...
    def explain(self, *args: Any, **kwargs: Any) -> Any: ...
    def join(self, *args: Any, **kwargs: Any) -> Any: ...


@pytest.mark.filterwarnings("ignore:.*You passed a.*:UserWarning")
def test_is_into_lazyframe(constructor: Constructor) -> None:
    native_frame = constructor(data)
    nw_frame = nw.from_native(native_frame)
    nw_v1_frame = nw_v1.from_native(native_frame)
    nw_v2_frame = nw_v2.from_native(native_frame)

    result = not any(x in str(constructor) for x in EAGER_CONSTRUCTOR_NAMES)
    assert is_into_lazyframe(native_frame) == result
    assert is_into_lazyframe(nw_frame) == result

    result_v1 = not any(x in str(constructor) for x in V1_INTO_DATAFRAMES)
    assert v1_is_into_lazyframe(native_frame) == result_v1
    assert v1_is_into_lazyframe(nw_v2_frame) is False

    result_v2 = not any(x in str(constructor) for x in EAGER_CONSTRUCTOR_NAMES)
    assert v2_is_into_lazyframe(native_frame) == result_v2
    assert v2_is_into_lazyframe(nw_v2_frame) == result_v2
    assert v2_is_into_lazyframe(nw_v1_frame) is False

    assert is_into_lazyframe(nw_v1_frame) == result_v1
    assert is_into_lazyframe(nw_v2_frame) == result_v2
    assert not v1_is_into_lazyframe(nw_frame)
    assert not v2_is_into_lazyframe(nw_frame)


def test_is_into_lazyframe_other() -> None:
    pytest.importorskip("numpy")
    import numpy as np

    assert is_into_lazyframe(DictLazyFrame(data))
    assert not is_into_lazyframe(np.array([[1, 4], [2, 5], [3, 6]]))
    assert not is_into_lazyframe(data)

    assert v1_is_into_lazyframe(DictLazyFrame(data))
    assert not v1_is_into_lazyframe(np.array([[1, 4], [2, 5], [3, 6]]))
    assert not v1_is_into_lazyframe(data)

    assert v2_is_into_lazyframe(DictLazyFrame(data))
    assert not v2_is_into_lazyframe(np.array([[1, 4], [2, 5], [3, 6]]))
    assert not v2_is_into_lazyframe(data)
