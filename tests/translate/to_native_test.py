from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from tests.utils import ConstructorEager


@pytest.mark.parametrize(
    ("method", "pass_through", "context"),
    [
        ("head", False, does_not_raise()),
        ("head", True, does_not_raise()),
        ("to_numpy", True, does_not_raise()),
        (
            "to_numpy",
            False,
            pytest.raises(TypeError, match="Expected Narwhals object, got"),
        ),
    ],
)
def test_to_native(
    constructor_eager: ConstructorEager, method: str, *, pass_through: bool, context: Any
) -> None:
    df = nw.from_native(constructor_eager({"a": [1, 2, 3]}))

    with context:
        nw.to_native(getattr(df, method)(), pass_through=pass_through)

    s = df["a"]

    with context:
        nw.to_native(getattr(s, method)(), pass_through=pass_through)
