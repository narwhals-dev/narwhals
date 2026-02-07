from __future__ import annotations

import re
from typing import Any

import pytest

import narwhals as nw
from tests.utils import ConstructorEager, assert_equal_data


@pytest.mark.parametrize(
    "data", [[1, 2, 3], ["foo", "bar", "baz"], [0.1, 0.2, 0.3], [True, False, True]]
)
@pytest.mark.parametrize("n", [0, 1, 10])
def test_clear(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    data: list[Any],
    n: int,
) -> None:
    if n > 0 and (
        any(
            x in str(constructor_eager)
            for x in ("cudf", "pandas_constructor", "modin_constructor")
        )
        or ("pandas_nullable" in str(constructor_eager) and isinstance(data[0], str))
    ):
        reason = "NotImplementedError"
        request.applymarker(pytest.mark.xfail(reason))

    s = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]

    s_clear = s.clear(n=n)
    assert len(s_clear) == n
    assert s.dtype == s_clear.dtype

    assert_equal_data({"a": s_clear}, {"a": [None] * n})


@pytest.mark.parametrize("n", [-1, "foo", 2.0, 1 + 1j])
def test_clear_exception(constructor_eager: ConstructorEager, n: Any) -> None:
    data = {"a": [1, 2, 3]}
    df = nw.from_native(constructor_eager(data), eager_only=True)["a"]

    msg = re.escape(f"`n` should be an integer >= 0, got {n}")
    with pytest.raises((TypeError, ValueError), match=msg):
        df.clear(n=n)
