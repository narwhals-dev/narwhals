from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals.exceptions import InvalidOperationError
from tests.utils import POLARS_VERSION

if TYPE_CHECKING:
    from tests.utils import ConstructorEager

data = [100, 200, None]


@pytest.mark.parametrize(
    ("other", "expected"), [(100, True), (None, True), (1, False), (100.314, False)]
)
def test_contains(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    other: float | None,
    *,
    expected: bool,
) -> None:
    if (
        isinstance(other, float)
        and "polars" in str(constructor_eager)
        and POLARS_VERSION >= (2,)
    ):  # pragma: no cover
        reason = "Polars>=2.0 rejects operands which can't be coerced losslessly"
        request.applymarker(pytest.mark.xfail(reason=reason))

    s = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]

    assert (other in s) == expected


@pytest.mark.parametrize("other", ["foo", [1, 2, 3]])
def test_contains_invalid_type(
    request: pytest.FixtureRequest, constructor_eager: ConstructorEager, other: Any
) -> None:
    if "polars" not in str(constructor_eager) and "pyarrow_table" not in str(
        constructor_eager
    ):
        request.applymarker(pytest.mark.xfail)

    s = nw.from_native(constructor_eager({"a": data}), eager_only=True)["a"]

    with pytest.raises(InvalidOperationError):
        _ = other in s
