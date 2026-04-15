from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from narwhals.testing.typing import ConstructorEager


def test_new_series(constructor_eager: ConstructorEager) -> None:
    s = nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)["a"]
    result = nw.new_series("b", [4, 1, 2], backend=nw.get_native_namespace(s))
    expected = {"b": [4, 1, 2]}
    # all supported libraries auto-infer this to be int64, we can always special-case
    # something different if necessary
    assert result.dtype == nw.Int64
    assert_equal_data(result.to_frame(), expected)

    result = nw.new_series("b", [4, 1, 2], nw.Int32, backend=nw.get_native_namespace(s))
    expected = {"b": [4, 1, 2]}
    assert result.dtype == nw.Int32
    assert_equal_data(result.to_frame(), expected)


def test_new_series_dask() -> None:
    pytest.importorskip("dask")
    import dask.dataframe as dd
    import pandas as pd

    df = nw.from_native(dd.from_pandas(pd.DataFrame({"a": [1, 2, 3]})))
    with pytest.raises(ValueError, match="lazy-only"):
        nw.new_series("a", [1, 2, 3], backend=nw.get_native_namespace(df))
