from __future__ import annotations

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


def test_clear_negative(constructor_eager: ConstructorEager) -> None:
    n = -1
    data = {"a": [1, 2, 3]}
    series = nw.from_native(constructor_eager(data), eager_only=True)["a"]

    msg = f"`n` should be greater than or equal to 0, got {n}"
    with pytest.raises(ValueError, match=msg):
        series.clear(n=n)


@pytest.mark.parametrize("n", ["foo", 2.0, 1 + 1j])
def test_clear_non_integer(constructor_eager: ConstructorEager, n: Any) -> None:
    data = {"a": [1, 2, 3]}
    df = nw.from_native(constructor_eager(data), eager_only=True)["a"]

    msg = "`n` should be an integer, got type"
    with pytest.raises(TypeError, match=msg):
        df.clear(n=n)
