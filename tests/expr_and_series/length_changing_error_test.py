from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from tests.utils import Constructor

data = {"a": [1, 1, None, 2], "b": [1, 2, 3, 4]}


@pytest.mark.parametrize(
    "expr",
    [nw.col("a").unique(), nw.col("a").drop_nulls(), nw.col("a").filter(nw.col("b") > 1)],
    ids=["unique", "drop_nulls", "filter"],
)
def test_length_changing_error_message(constructor: Constructor, expr: nw.Expr) -> None:
    # Lazy backends should explain *why* length-changing expressions are unavailable,
    # rather than just reporting them as not implemented.
    # See https://github.com/narwhals-dev/narwhals/issues/3898.
    df = nw.from_native(constructor(data))
    if not isinstance(df, nw.LazyFrame):
        pytest.skip(reason="Length-changing expressions are supported for eager backends")
    with pytest.raises((InvalidOperationError, NotImplementedError)) as exc_info:
        df.select(expr)
    msg = str(exc_info.value)
    assert "Length-changing expressions" in msg
    assert "Hint" in msg
