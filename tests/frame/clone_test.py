from typing import Any

import pytest

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_clone(constructor_with_lazy: Any) -> None:
    if "pyarrow_table" in str(constructor_with_lazy):
        pytest.xfail()

    expected = {"a": [1, 2], "b": [3, 4]}
    df = nw.from_native(constructor_with_lazy(expected))
    df_clone = df.clone()
    assert df is not df_clone
    assert df._compliant_frame is not df_clone._compliant_frame
    compare_dicts(df_clone, expected)
