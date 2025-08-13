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


def _pytest_assertion_error(detail: str) -> Any:
    return pytest.raises(AssertionError, match=rf"Series are different \({detail}\)")


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

    with _pytest_assertion_error("implementation mismatch"):
        assert_series_equal(left, right)  # type: ignore[arg-type]


def test_raise_length_mismatch(constructor_eager: ConstructorEager) -> None:
    left = nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)["a"]
    right = left.head(2)
    msg = r"Series are different \(length mismatch\)"
    with pytest.raises(AssertionError, match=msg):
        assert_series_equal(left, right)


@pytest.mark.parametrize(
    ("check_dtypes", "context"),
    [(False, nullcontext()), (True, _pytest_assertion_error("dtype mismatch"))],
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
    [(False, nullcontext()), (True, _pytest_assertion_error("name mismatch"))],
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
    [(False, nullcontext()), (True, _pytest_assertion_error("exact value mismatch"))],
)
def test_non_nested_check_order(
    constructor_eager: ConstructorEager, *, check_order: bool, context: Any
) -> None:
    data = {"left": ["a", "b", "c"], "right": ["b", "c", "a"]}
    frame = nw.from_native(constructor_eager(data), eager_only=True)
    left, right = frame["left"], frame["right"]
    with context:
        assert_series_equal(left, right, check_order=check_order, check_names=False)


@pytest.mark.parametrize(
    "_data",
    [
        {"left": ["x", None, None], "right": ["x", None, "x"]},
        {"left": ["x", None, None], "right": [None, None, "x"]},
    ],
)
def test_null_mismatch(constructor_eager: ConstructorEager, _data: Data) -> None:
    frame = nw.from_native(constructor_eager(_data), eager_only=True)
    left, right = frame["left"], frame["right"]
    with _pytest_assertion_error("null value mismatch"):
        assert_series_equal(left, right, check_names=False)


@pytest.mark.parametrize(
    ("check_exact", "abs_tol", "rel_tol", "context"),
    [
        (True, 1e-3, 1e-3, _pytest_assertion_error("exact value mismatch")),
        (False, 1e-3, 1e-3, _pytest_assertion_error("values not within tolerance")),
        (False, 2e-1, 2e-1, nullcontext()),
    ],
)
def test_numeric(
    constructor_eager: ConstructorEager,
    *,
    check_exact: bool,
    abs_tol: float,
    rel_tol: float,
    context: Any,
) -> None:
    data = {
        "left": [1.0, float("nan"), float("inf"), None, 1.1],
        "right": [1.01, float("nan"), float("inf"), None, 1.11],
    }

    frame = nw.from_native(constructor_eager(data), eager_only=True)
    left, right = frame["left"], frame["right"]
    with context:
        assert_series_equal(
            left,
            right,
            check_names=False,
            check_exact=check_exact,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )


@pytest.mark.parametrize(
    ("l_vals", "r_vals", "check_exact", "context", "dtype"),
    [
        (
            [["foo", "bar"]],
            [["foo", None]],
            True,
            _pytest_assertion_error("nested value mismatch"),
            nw.List(nw.String()),
        ),
        (
            [["foo", "bar"]],
            [["foo", None]],
            True,
            _pytest_assertion_error("nested value mismatch"),
            nw.Array(nw.String(), 2),
        ),
        (
            [[0.0, 0.1]],
            [[0.1, 0.1]],
            True,
            _pytest_assertion_error("nested value mismatch"),
            nw.List(nw.Float32()),
        ),
        (
            [[0.0, 0.1]],
            [[0.1, 0.1]],
            True,
            _pytest_assertion_error("nested value mismatch"),
            nw.Array(nw.Float32(), 2),
        ),
        ([[0.0, 1e-10]], [[1e-10, 0.0]], False, nullcontext(), nw.List(nw.Float64())),
        ([[0.0, 1e-10]], [[1e-10, 0.0]], False, nullcontext(), nw.Array(nw.Float64(), 2)),
    ],
)
def test_list_like(
    constructor_eager: ConstructorEager,
    l_vals: list[list[Any]],
    r_vals: list[list[Any]],
    *,
    check_exact: bool,
    context: Any,
    dtype: nw.dtypes.DType,
) -> None:
    data = {"left": l_vals, "right": r_vals}
    frame = nw.from_native(constructor_eager(data), eager_only=True).select(
        nw.all().cast(dtype)
    )
    left, right = frame["left"], frame["right"]
    with context:
        assert_series_equal(left, right, check_names=False, check_exact=check_exact)


@pytest.mark.parametrize(
    ("l_vals", "r_vals", "check_exact", "context"),
    [
        (
            [{"a": 0.0, "b": ["orca"]}, None],
            [{"a": 1e-10, "b": ["orca"]}, None],
            True,
            _pytest_assertion_error("exact value mismatch"),
        ),
        (
            [{"a": 0.0, "b": ["beluga"]}, None],
            [{"a": 0.0, "b": ["orca"]}, None],
            False,
            _pytest_assertion_error("exact value mismatch"),
        ),
        (
            [{"a": 0.0, "b": ["orca"]}, None],
            [{"a": 1e-10, "b": ["orca"]}, None],
            False,
            nullcontext(),
        ),
    ],
)
def test_struct(
    constructor_eager: ConstructorEager,
    l_vals: list[dict[str, Any]],
    r_vals: list[dict[str, Any]],
    *,
    check_exact: bool,
    context: Any,
) -> None:
    dtype = nw.Struct({"a": nw.Float32(), "b": nw.List(nw.String())})
    data = {"left": l_vals, "right": r_vals}
    frame = nw.from_native(constructor_eager(data), eager_only=True).select(
        nw.all().cast(dtype)
    )
    left, right = frame["left"], frame["right"]
    with context:
        assert_series_equal(left, right, check_names=False, check_exact=check_exact)
