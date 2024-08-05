from typing import Any

import pytest

import narwhals.stable.v1 as nw


@pytest.mark.filterwarnings("ignore:Determining|Resolving.*")
def test_columns(constructor: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8, 9]}
    df = nw.from_native(constructor(data))
    result = df.columns
    expected = ["a", "b", "z"]
    assert result == expected
