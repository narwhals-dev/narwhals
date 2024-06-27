from typing import Any

import narwhals.stable.v1 as nw
from tests.utils import compare_dicts

data = {
    "a": ["foo", "bars"],
    "ab": ["foo", "bars"],
}


def test_pipe(constructor: Any) -> None:
    df = nw.from_native(constructor(data))
    columns = df.lazy().collect().columns
    result = df.pipe(lambda _df: _df.select([x for x in columns if len(x) == 2]))
    expected = {"ab": ["foo", "bars"]}
    compare_dicts(result, expected)
    result = df.lazy().pipe(lambda _df: _df.select([x for x in columns if len(x) == 2]))
    compare_dicts(result, expected)
