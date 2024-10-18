from __future__ import annotations

import pytest

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import compare_dicts

data = {"a": list(range(10))}


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("offset", [1, 2, 3])
def test_gather_every(constructor: Constructor, n: int, offset: int) -> None:
    df = nw.from_native(constructor(data))
    result = df.gather_every(n=n, offset=offset)
    expected = {"a": data["a"][offset::n]}
    compare_dicts(result, expected)
