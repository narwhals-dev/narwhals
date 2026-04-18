from __future__ import annotations

import narwhals as nw
from tests.utils import Constructor, assert_equal_data

data = {"a": ["foo", "bars"], "ab": ["foo", "bars"]}


def test_pipe(nw_frame_constructor: Constructor) -> None:
    df = nw.from_native(nw_frame_constructor(data))
    columns = df.collect_schema().names()
    result = df.pipe(lambda _df: _df.select([x for x in columns if len(x) == 2]))
    expected = {"ab": ["foo", "bars"]}
    assert_equal_data(result, expected)
