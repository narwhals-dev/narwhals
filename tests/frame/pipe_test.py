from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
from tests.utils import assert_equal_data

if TYPE_CHECKING:
    from narwhals.testing.typing import Constructor

data = {"a": ["foo", "bars"], "ab": ["foo", "bars"]}


def test_pipe(constructor: Constructor) -> None:
    df = nw.from_native(constructor(data))
    columns = df.collect_schema().names()
    result = df.pipe(lambda _df: _df.select([x for x in columns if len(x) == 2]))
    expected = {"ab": ["foo", "bars"]}
    assert_equal_data(result, expected)
