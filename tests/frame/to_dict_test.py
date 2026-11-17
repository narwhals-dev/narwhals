from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data


@pytest.mark.filterwarnings(
    "ignore:.*all arguments of to_dict except for the argument:FutureWarning"
)
def test_to_dict(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "c": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.to_dict(as_series=False)
    assert result == data


def test_to_dict_as_series(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "c": [7.0, 8.0, 9.0]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.to_dict(as_series=True)
    assert isinstance(result["a"], nw.Series)
    assert isinstance(result["b"], nw.Series)
    assert isinstance(result["c"], nw.Series)
    assert_equal_data(result, data)


def test_to_dict_pyarrow_backed_pandas_nulls() -> None:
    pd = pytest.importorskip("pandas")
    pa = pytest.importorskip("pyarrow")
    tbl = pa.table({"i": [1, None, 3], "f": [1.5, float("nan"), None]})
    native = tbl.to_pandas(types_mapper=pd.ArrowDtype)
    df = nw.from_native(native, eager_only=True)
    result = df.to_dict(as_series=False)
    assert result["i"] == [1, None, 3]
    assert result["f"][0] == 1.5
    assert result["f"][1] != result["f"][1]  # NaN
    assert result["f"][2] is None
