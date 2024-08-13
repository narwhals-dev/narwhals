from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts


def test_new_series(constructor_eager: Any) -> None:
    s = nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)["a"]
    result = nw.new_series("b", [4, 1, 2], native_namespace=nw.get_native_namespace(s))
    expected = {"b": [4, 1, 2]}
    # all supported libraries auto-infer this to be int64, we can always special-case
    # something different if necessary
    assert result.dtype == nw.Int64
    compare_dicts(result.to_frame(), expected)

    result = nw.new_series(
        "b", [4, 1, 2], nw.Int32, native_namespace=nw.get_native_namespace(s)
    )
    expected = {"b": [4, 1, 2]}
    # all supported libraries auto-infer this to be int64, we can always special-case
    # something different if necessary
    assert result.dtype == nw.Int32
    compare_dicts(result.to_frame(), expected)
