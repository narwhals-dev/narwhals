from __future__ import annotations

import re
from contextlib import AbstractContextManager, nullcontext as does_not_raise
from typing import TYPE_CHECKING, Any, Callable

import pytest

import narwhals as nw
from narwhals.testing import assert_series_equal
from tests.utils import PANDAS_VERSION, POLARS_VERSION, PYARROW_VERSION

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals.typing import IntoSchema, IntoSeriesT
    from tests.conftest import Data
    from tests.utils import ConstructorEager

    SetupFn: TypeAlias = Callable[[nw.Series[Any]], tuple[nw.Series[Any], nw.Series[Any]]]


def _assertion_error(detail: str) -> pytest.RaisesExc:
    return pytest.raises(
        AssertionError, match=re.escape(f"Series are different ({detail})")
    )


def series_from_native(native: IntoSeriesT) -> nw.Series[IntoSeriesT]:
    return nw.from_native(native, series_only=True)


def test_self_equal(
    constructor_eager: ConstructorEager, testing_data: Data, testing_schema: IntoSchema
) -> None:
    """Test that a series is equal to itself, including nested dtypes with nulls."""
    if "pandas" in str(constructor_eager):
        if PANDAS_VERSION < (2, 2):  # pragma: no cover
            reason = "Pandas too old for nested dtypes"
            pytest.skip(reason=reason)
        pytest.importorskip("pyarrow")

    if "pyarrow_table" in str(constructor_eager) and PYARROW_VERSION < (
        15,
        0,
    ):  # pragma: no cover
        reason = (
            "pyarrow.lib.ArrowNotImplementedError: Unsupported cast from string to "
            "dictionary using function cast_dictionary"
        )
        pytest.skip(reason=reason)

    if "pyarrow_table" in str(constructor_eager):
        # Replace Enum with Categorical, since Pyarrow does not support Enum
        schema = {**testing_schema, "enum": nw.Categorical()}
    else:
        schema = dict(testing_schema)  # make a copy
    df = nw.from_native(constructor_eager(testing_data), eager_only=True)
    for name, dtype in schema.items():
        assert_series_equal(df[name].cast(dtype), df[name].cast(dtype))


def test_implementation_mismatch() -> None:
    """Test that different implementations raise an error."""
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    import pandas as pd
    import pyarrow as pa

    with _assertion_error("implementation mismatch"):
        assert_series_equal(
            series_from_native(pd.Series([1])),
            series_from_native(pa.chunked_array([[2]])),  # type: ignore[misc] # pyright: ignore[reportArgumentType]
        )


@pytest.mark.parametrize(
    ("setup_fn", "error_msg"),
    [
        (lambda s: (s, s.head(2)), "length mismatch"),
        (lambda s: (s.cast(nw.UInt32()), s.cast(nw.Int64())), "dtype mismatch"),
        (lambda s: (s.rename("foo"), s.rename("bar")), "name mismatch"),
    ],
)
def test_metadata_checks(
    constructor_eager: ConstructorEager, setup_fn: SetupFn, error_msg: str
) -> None:
    """Test metadata validation (length, dtype, name)."""
    series = nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)["a"]
    left, right = setup_fn(series)

    with _assertion_error(error_msg):
        assert_series_equal(left, right)


@pytest.mark.parametrize(
    ("setup_fn", "error_msg", "check_dtypes", "check_names"),
    [
        (lambda s: (s, s.cast(nw.UInt32())), "dtype mismatch", True, False),
        (lambda s: (s, s.cast(nw.UInt32()).rename("baz")), "dtype mismatch", True, True),
        (lambda s: (s, s.rename("baz")), "name mismatch", False, True),
    ],
)
def test_metadata_checks_with_flags(
    constructor_eager: ConstructorEager,
    setup_fn: SetupFn,
    error_msg: str,
    *,
    check_dtypes: bool,
    check_names: bool,
) -> None:
    """Test the effect of check_dtypes and check_names flags."""
    series = nw.from_native(constructor_eager({"a": [1, 2, 3]}), eager_only=True)["a"]
    left, right = setup_fn(series)

    with _assertion_error(error_msg):
        assert_series_equal(
            left, right, check_dtypes=check_dtypes, check_names=check_names
        )

    assert_series_equal(left, right, check_dtypes=False, check_names=False)


@pytest.mark.parametrize(
    ("dtype", "check_order", "context"),
    [
        (nw.List(nw.Int32()), False, pytest.raises(NotImplementedError)),
        (nw.List(nw.Int32()), True, does_not_raise()),
        (nw.Int32(), False, does_not_raise()),
        (nw.Int32(), True, does_not_raise()),
    ],
)
def test_check_order(
    constructor_eager: ConstructorEager,
    dtype: nw.dtypes.DType,
    *,
    check_order: bool,
    context: AbstractContextManager[Any],
) -> None:
    """Test check_order behavior with nested and simple data."""
    if "pandas" in str(constructor_eager):
        if PANDAS_VERSION < (2, 2) and dtype.is_nested():  # pragma: no cover
            reason = "Pandas too old for nested dtypes"
            pytest.skip(reason=reason)
        pytest.importorskip("pyarrow")

    data: list[Any] = [[1, 2, 3]] if dtype.is_nested() else [1, 2, 3]
    frame = nw.from_native(constructor_eager({"a": data}), eager_only=True)
    left = right = frame["a"].cast(dtype)

    with context:
        assert_series_equal(left, right, check_order=check_order, check_names=False)


@pytest.mark.parametrize(
    "null_data",
    [
        {"left": ["x", "y", None], "right": ["x", None, "y"]},  # Different null position
        {"left": ["x", None, None], "right": [None, "x", "y"]},  # Different null counts
    ],
)
def test_null_mismatch(constructor_eager: ConstructorEager, null_data: Data) -> None:
    """Test null value mismatch detection."""
    frame = nw.from_native(constructor_eager(null_data), eager_only=True)
    left, right = frame["left"], frame["right"]
    with _assertion_error("null value mismatch"):
        assert_series_equal(left, right, check_names=False)


@pytest.mark.parametrize(
    ("check_exact", "abs_tol", "rel_tol", "context"),
    [
        (True, 1e-3, 1e-3, _assertion_error("exact value mismatch")),
        (False, 1e-3, 1e-3, _assertion_error("values not within tolerance")),
        (False, 2e-1, 2e-1, does_not_raise()),
    ],
)
def test_numeric(
    constructor_eager: ConstructorEager,
    *,
    check_exact: bool,
    abs_tol: float,
    rel_tol: float,
    context: AbstractContextManager[Any],
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
            _assertion_error("nested value mismatch"),
            nw.List(nw.String()),
        ),
        (
            [["foo", "bar"]],
            [["foo", None]],
            True,
            _assertion_error("nested value mismatch"),
            nw.Array(nw.String(), 2),
        ),
        (
            [[0.0, 0.1]],
            [[0.1, 0.1]],
            True,
            _assertion_error("nested value mismatch"),
            nw.List(nw.Float32()),
        ),
        (
            [[0.0, 0.1]],
            [[0.1, 0.1]],
            True,
            _assertion_error("nested value mismatch"),
            nw.Array(nw.Float32(), 2),
        ),
        ([[0.0, 1e-10]], [[1e-10, 0.0]], False, does_not_raise(), nw.List(nw.Float64())),
        (
            [[0.0, 1e-10]],
            [[1e-10, 0.0]],
            False,
            does_not_raise(),
            nw.Array(nw.Float64(), 2),
        ),
    ],
)
def test_list_like(
    constructor_eager: ConstructorEager,
    l_vals: list[list[Any]],
    r_vals: list[list[Any]],
    *,
    check_exact: bool,
    context: AbstractContextManager[Any],
    dtype: nw.dtypes.DType,
) -> None:
    if "pandas" in str(constructor_eager):
        if PANDAS_VERSION < (2, 2):  # pragma: no cover
            reason = "Pandas too old for nested dtypes"
            pytest.skip(reason=reason)
        pytest.importorskip("pyarrow")

    if (
        "pyarrow_table" in str(constructor_eager)
        and PYARROW_VERSION < (14, 0)
        and dtype == nw.Array
    ):  # pragma: no cover
        reason = (
            "pyarrow.lib.ArrowNotImplementedError: Unsupported cast from "
            "list<item: string> to fixed_size_list using function cast_fixed_size_list"
        )
        pytest.skip(reason=reason)

    data = {"left": l_vals, "right": r_vals}
    frame = nw.from_native(constructor_eager(data), eager_only=True)
    left, right = frame["left"].cast(dtype), frame["right"].cast(dtype)
    with context:
        assert_series_equal(left, right, check_names=False, check_exact=check_exact)


@pytest.mark.parametrize(
    ("l_vals", "r_vals", "check_exact", "context"),
    [
        (
            [{"a": 0.0, "b": ["orca"]}, None],
            [{"a": 1e-10, "b": ["orca"]}, None],
            True,
            _assertion_error("exact value mismatch"),
        ),
        (
            [{"a": 0.0, "b": ["beluga"]}, None],
            [{"a": 0.0, "b": ["orca"]}, None],
            False,
            _assertion_error("exact value mismatch"),
        ),
        (
            [{"a": 0.0, "b": ["orca"]}, None],
            [{"a": 1e-10, "b": ["orca"]}, None],
            False,
            does_not_raise(),
        ),
    ],
)
def test_struct(
    constructor_eager: ConstructorEager,
    l_vals: list[dict[str, Any]],
    r_vals: list[dict[str, Any]],
    *,
    check_exact: bool,
    context: AbstractContextManager[Any],
) -> None:
    if "pandas" in str(constructor_eager):
        if PANDAS_VERSION < (2, 2):  # pragma: no cover
            reason = "Pandas too old for nested dtypes"
            pytest.skip(reason=reason)
        pytest.importorskip("pyarrow")

    dtype = nw.Struct({"a": nw.Float32(), "b": nw.List(nw.String())})
    data = {"left": l_vals, "right": r_vals}
    frame = nw.from_native(constructor_eager(data), eager_only=True)
    left, right = frame["left"].cast(dtype), frame["right"].cast(dtype)
    with context:
        assert_series_equal(left, right, check_names=False, check_exact=check_exact)


def test_non_nw_series() -> None:
    pytest.importorskip("pandas")

    import pandas as pd

    with pytest.raises(
        TypeError, match=re.escape("Expected `narwhals.Series` instance, found")
    ):
        assert_series_equal(
            left=pd.Series([1]),  # type: ignore[arg-type]
            right=pd.Series([2]),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("categorical_as_str", "context"),
    [
        (True, does_not_raise()),
        (
            False,
            pytest.raises(
                AssertionError,
                match="Cannot compare categoricals coming from different sources",
            ),
        ),
    ],
)
def test_categorical_as_str(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    *,
    categorical_as_str: bool,
    context: AbstractContextManager[Any],
) -> None:
    if (
        "polars" in str(constructor_eager)
        and POLARS_VERSION >= (1, 32)
        and not categorical_as_str
    ):
        # https://github.com/pola-rs/polars/pull/23016 removed StringCache, it still
        # exists but it does nothing in python.
        request.applymarker(pytest.mark.xfail)

    if "pyarrow_table" in str(constructor_eager) and not categorical_as_str:
        # pyarrow dictionary dtype compares values, not the encoding.
        request.applymarker(pytest.mark.xfail)

    if "pyarrow_table" in str(constructor_eager) and PYARROW_VERSION < (
        15,
        0,
    ):  # pragma: no cover
        reason = (
            "pyarrow.lib.ArrowNotImplementedError: Unsupported cast from string to "
            "dictionary using function cast_dictionary"
        )
        pytest.skip(reason=reason)

    data = {
        "left": ["beluga", "dolphin", "narwhal", "orca"],
        "right": ["unicorn", "orca", "narwhal", "orca"],
    }
    frame = nw.from_native(constructor_eager(data), eager_only=True)
    left = frame["left"].cast(nw.Categorical())[2:]
    right = frame["right"].cast(nw.Categorical())[2:]

    with context:
        assert_series_equal(
            left, right, check_names=False, categorical_as_str=categorical_as_str
        )
