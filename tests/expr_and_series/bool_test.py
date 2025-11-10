from __future__ import annotations

import pytest

import narwhals as nw


def test_expr_bool_raises() -> None:
    expr = nw.col("a")
    with pytest.raises(TypeError, match=r"the truth value of .* is ambiguous"):
        bool(expr)


def test_series_bool_raises() -> None:
    series = object.__new__(nw.Series)
    with pytest.raises(TypeError, match=r"the truth value of .* is ambiguous"):
        bool(series)
