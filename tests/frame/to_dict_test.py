from typing import Any

import narwhals.stable.v1 as nw


def test_to_dict(constructor: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "c": [7.0, 8, 9]}
    df = nw.from_native(constructor(data), eager_only=True)
    result = df.to_dict(as_series=False)
    assert result == data


def test_to_dict_as_series(constructor: Any) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "c": [7.0, 8, 9]}
    df = nw.from_native(constructor(data), eager_only=True)
    result = df.to_dict(as_series=True)
    assert isinstance(result["a"], nw.Series)
    assert isinstance(result["b"], nw.Series)
    assert isinstance(result["c"], nw.Series)
    assert {key: value.to_list() for key, value in result.items()} == data
