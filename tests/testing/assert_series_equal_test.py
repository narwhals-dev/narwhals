from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals.testing import assert_series_equal

if TYPE_CHECKING:
    from narwhals.typing import IntoSchema
    from tests.conftest import Data
    from tests.utils import ConstructorEager


def test_self_equal(
    constructor_eager: ConstructorEager, data: Data, schema: IntoSchema
) -> None:
    """This test exists for validate that nested dtypes are checked correctly, including nulls."""
    if "pyarrow_table" in str(constructor_eager):
        # Pyarrow does not support Enum
        schema = {**schema, "enum": nw.Categorical()}

    df = nw.from_native(constructor_eager(data), eager_only=True).select(
        nw.col(name).cast(dtype) for name, dtype in schema.items()
    )
    for name in schema:
        assert_series_equal(df[name], df[name])


def test_raise_impl_mismatch() -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    import pandas as pd
    import pyarrow as pa

    data = [1, 2, 3]
    left, right = pd.Series(data), pa.chunked_array([data])

    msg = r"Series are different \(implementation mismatch\)"
    with pytest.raises(AssertionError, match=msg):
        assert_series_equal(left, right)  # type: ignore[arg-type]


def test_raise_length_mismatch(constructor_eager: ConstructorEager) -> None:
    left = nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)["a"]
    right = left.head(2)
    msg = r"Series are different \(length mismatch\)"
    with pytest.raises(AssertionError, match=msg):
        assert_series_equal(left, right)


@pytest.mark.parametrize(
    ("check_dtypes", "context"),
    [
        (False, nullcontext()),
        (
            True,
            pytest.raises(
                AssertionError, match=r"Series are different \(dtype mismatch\)"
            ),
        ),
    ],
)
def test_check_dtypes(
    constructor_eager: ConstructorEager, *, check_dtypes: bool, context: Any
) -> None:
    left = nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)["a"]
    with context:
        assert_series_equal(
            left.cast(nw.UInt32()), left.cast(nw.Int64()), check_dtypes=check_dtypes
        )


@pytest.mark.parametrize(
    ("check_names", "context"),
    [
        (False, nullcontext()),
        (
            True,
            pytest.raises(
                AssertionError, match=r"Series are different \(name mismatch\)"
            ),
        ),
    ],
)
def test_check_names(
    constructor_eager: ConstructorEager, *, check_names: bool, context: Any
) -> None:
    left = nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)["a"]
    with context:
        assert_series_equal(
            left.rename("foo"), left.rename("bar"), check_names=check_names
        )


@pytest.mark.parametrize(
    ("check_order", "context"),
    [(True, nullcontext()), (False, pytest.raises(NotImplementedError))],
)
def test_nested_check_order(
    constructor_eager: ConstructorEager, *, check_order: bool, context: Any
) -> None:
    left = nw.from_native(constructor_eager({"a": [[1, 2, 3]]}), eager_only=True)[
        "a"
    ].cast(nw.List(nw.Int32()))
    with context:
        assert_series_equal(left, left, check_order=check_order)


@pytest.mark.parametrize(
    ("check_order", "context"),
    [
        (False, nullcontext()),
        (
            True,
            pytest.raises(
                AssertionError, match=r"Series are different \(exact value mismatch\)"
            ),
        ),
    ],
)
def test_non_nested_check_order(
    constructor_eager: ConstructorEager, *, check_order: bool, context: Any
) -> None:
    data = {"left": ["a", "b", "c"], "right": ["b", "c", "a"]}
    frame = nw.from_native(constructor_eager(data), eager_only=True)
    left, right = frame["left"], frame["right"]
    with context:
        assert_series_equal(left, right, check_order=check_order, check_names=False)


def test_null_mismatch(constructor_eager: ConstructorEager) -> None: ...
