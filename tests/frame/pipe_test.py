from __future__ import annotations

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import compare_dicts

data = {
    "a": ["foo", "bars"],
    "ab": ["foo", "bars"],
}


def test_pipe(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    columns = df.collect_schema().names()
    result = df.pipe(lambda _df: _df.select([x for x in columns if len(x) == 2]))
    expected = {"ab": ["foo", "bars"]}
    compare_dicts(result, expected)
