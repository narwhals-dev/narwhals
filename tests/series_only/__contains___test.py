from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError

if TYPE_CHECKING:
    from tests.utils import ConstructorEager

# NOTE: floats, as pandas represents `[100, 200, None]` as `Float64`, and Narwhals
# only compares values whose dtype is compatible with the series' one.
data = [100.0, 200.0, None]


@pytest.mark.parametrize(
    ("other", "expected"), [(100.0, True), (None, True), (1.0, False)]
)
def test_contains(
    constructor_eager: ConstructorEager, other: float | None, *, expected: bool
) -> None:
    s = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]

    assert (other in s) == expected


@pytest.mark.parametrize(("other", "expected"), [(100, True), (1, False)])
def test_contains_integer(
    constructor_eager: ConstructorEager, other: int, *, expected: bool
) -> None:
    s = nw.from_native(constructor_eager({"a": [100, 200]}), eager_only=True)["a"]

    assert (other in s) == expected


@pytest.mark.parametrize("other", [100.314, "foo", [1, 2, 3]])
def test_contains_invalid_type(constructor_eager: ConstructorEager, other: Any) -> None:
    # Narwhals follows Polars: values are only compared against a column when doing
    # so is lossless, rather than silently coercing both to a common dtype.
    s = nw.from_native(constructor_eager({"a": [100, 200]}), eager_only=True)["a"]

    with pytest.raises(InvalidOperationError, match="cannot check for"):
        _ = other in s


def test_contains_unsupported_value(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager
) -> None:
    # Narwhals can't tell whether an arbitrary object is comparable against a
    # column, so it's left to the backend to accept or reject it.
    if "polars" not in str(constructor_eager) and "pyarrow_table" not in str(
        constructor_eager
    ):
        request.applymarker(pytest.mark.xfail)

    s = nw.from_native(constructor_eager({"a": [100, 200]}), eager_only=True)["a"]

    with pytest.raises((InvalidOperationError, TypeError)):
        _ = object() in s


def test_contains_integer_in_float_series(constructor_eager: ConstructorEager) -> None:
    s = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]

    # NOTE: the message comes from Polars itself on `polars>=2.0`.
    with pytest.raises(InvalidOperationError, match="cannot check for"):
        _ = 100 in s
