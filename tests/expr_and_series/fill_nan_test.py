from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, assert_equal_data


def test_is_nan(constructor: Constructor) -> None:
    df = nw.from_native(
        constructor({"a": [1.1, 2.0, float("nan")], "b": [3.1, 4.0, None]})
    )
    result = df.select(nw.all().fill_nan(None))
    expected = {"a": [1.1, 2.0, None], "b": [3.1, 4.0, None]}
    assert_equal_data(result, expected)
    assert result.lazy().collect()["a"].null_count() == 1
    result = df.select(nw.all().fill_nan(3.0))
    if any(x in str(constructor) for x in ("pandas", "dask", "cudf", "modin")):
        # pandas doesn't distinguish nan vs null
        expected = {"a": [1.1, 2.0, 3.0], "b": [3.1, 4.0, 3.0]}
        assert int(result.lazy().collect()["b"].null_count()) == 0
    else:
        expected = {"a": [1.1, 2.0, 3.0], "b": [3.1, 4.0, None]}
        assert result.lazy().collect()["b"].null_count() == 1
    assert_equal_data(result, expected)
    assert result.lazy().collect()["a"].null_count() == 0
