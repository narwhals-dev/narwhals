from __future__ import annotations

import re
from contextlib import AbstractContextManager, nullcontext as does_not_raise
from typing import TYPE_CHECKING, Any, Callable

import pytest

import narwhals as nw
from narwhals.testing import assert_frame_equal
from narwhals.testing.asserts.frame import GUARANTEES_ROW_ORDER
from tests.utils import PANDAS_VERSION

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from narwhals.typing import IntoSchema
    from tests.conftest import Data
    from tests.utils import Constructor, ConstructorEager

    SetupFn: TypeAlias = Callable[[nw.Series[Any]], tuple[nw.Series[Any], nw.Series[Any]]]


def _assertion_error(detail: str) -> pytest.RaisesExc:
    msg = f"DataFrames are different ({detail})"
    return pytest.raises(AssertionError, match=re.escape(msg))


def test_check_narwhals_objects(constructor: Constructor) -> None:
    """Test that a type error is raised if the input is not a Narwhals object."""
    frame = constructor({"a": [1, 2, 3]})
    msg = re.escape(
        "Expected `narwhals.DataFrame` or `narwhals.LazyFrame` instance, found"
    )
    with pytest.raises(TypeError, match=msg):
        assert_frame_equal(frame, frame)  # type: ignore[arg-type]


def test_implementation_mismatch() -> None:
    """Test that different implementations raise an error."""
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    import pandas as pd
    import pyarrow as pa

    with _assertion_error("implementation mismatch"):
        assert_frame_equal(
            nw.from_native(pd.DataFrame({"a": [1]})),
            nw.from_native(pa.table({"a": [1]})),  # type: ignore[type-var] # pyright: ignore[reportArgumentType]
        )


def test_check_same_input_type(constructor_eager: ConstructorEager) -> None:
    """Test that left and right frames are either both eager or both lazy.

    NOTE: Use `constructor_eager` instead of `constructor` so that the roundtrip
        `.lazy().collect()` preserves the same implementation (and we raise the check after)
    """
    frame = nw.from_native(constructor_eager({"a": [1, 2, 3]}))

    msg = re.escape("inputs are different (unexpected input types)")
    with pytest.raises(AssertionError, match=msg):
        assert_frame_equal(frame.lazy(), frame.lazy().collect())


@pytest.mark.parametrize(
    ("left_schema", "right_schema", "check_dtypes", "check_column_order", "context"),
    [
        # Same order, same dtypes
        (
            {"a": nw.Int32(), "b": nw.Float32()},
            {"a": nw.Int32(), "b": nw.Float32()},
            True,
            True,
            does_not_raise(),
        ),
        # Same order, different dtypes
        (
            {"a": nw.Int32(), "b": nw.Float32()},
            {"a": nw.Int32(), "b": nw.Float64()},
            False,
            True,
            does_not_raise(),
        ),
        (
            {"a": nw.Int32(), "b": nw.Float32()},
            {"a": nw.Int32(), "b": nw.Float64()},
            True,
            True,
            _assertion_error("dtypes do not match"),
        ),
        # Different order, same dtype
        (
            {"a": nw.Int32(), "b": nw.Float32()},
            {"b": nw.Float32(), "a": nw.Int32()},
            True,
            False,
            does_not_raise(),
        ),
        (
            {"a": nw.Int32(), "b": nw.Float32()},
            {"b": nw.Float32(), "a": nw.Int32()},
            True,
            True,
            _assertion_error("columns are not in the same order"),
        ),
        # Different order, different dtype
        (
            {"a": nw.Int32(), "b": nw.Float32()},
            {"b": nw.Float64(), "a": nw.Int16()},
            False,
            False,
            does_not_raise(),
        ),
        (
            {"a": nw.Int32(), "b": nw.Float32()},
            {"b": nw.Float64(), "a": nw.Int16()},
            True,
            False,
            _assertion_error("dtypes do not match"),
        ),
        (
            {"a": nw.Int32(), "b": nw.Float32()},
            {"b": nw.Float64(), "a": nw.Int16()},
            False,
            True,
            _assertion_error("columns are not in the same order"),
        ),
        (
            {"a": nw.Int32(), "b": nw.Float32()},
            {"b": nw.Float64(), "a": nw.Int16()},
            True,
            True,
            _assertion_error("columns are not in the same order"),
        ),
        # Different columns (left not in right)
        (
            {"a": nw.Int32(), "z": nw.String()},
            {"b": nw.Float64()},
            True,
            True,
            _assertion_error("['a', 'z'] in left, but not in right"),
        ),
        # Different columns (right not in left)
        (
            {"z": nw.String()},
            {"z": nw.String(), "b": nw.Float64()},
            True,
            True,
            _assertion_error("['b'] in right, but not in left"),
        ),
    ],
)
def test_check_schema_mismatch(
    constructor: Constructor,
    left_schema: IntoSchema,
    right_schema: IntoSchema,
    *,
    check_dtypes: bool,
    check_column_order: bool,
    context: AbstractContextManager[Any],
) -> None:
    data = {"a": [1, 2, 3], "b": [4.5, 6.7, 8.9], "z": ["foo", "bar", "baz"]}
    left = nw.from_native(constructor(data)).select(
        nw.col(name).cast(dtype) for name, dtype in left_schema.items()
    )
    right = nw.from_native(constructor(data)).select(
        nw.col(name).cast(dtype) for name, dtype in right_schema.items()
    )

    with context:
        assert_frame_equal(
            left, right, check_column_order=check_column_order, check_dtypes=check_dtypes
        )


def test_height_mismatch(constructor: Constructor) -> None:
    left = nw.from_native(constructor({"a": [1, 2, 3]}))
    right = nw.from_native(constructor({"a": [1, 3]}))

    with _assertion_error("height (row count) mismatch"):
        assert_frame_equal(left, right)


@pytest.mark.parametrize("check_row_order", [True, False])
def test_check_row_order(
    constructor: Constructor, request: pytest.FixtureRequest, *, check_row_order: bool
) -> None:
    if "dask" in str(constructor):
        reason = "Unsupported List type"
        request.applymarker(pytest.mark.xfail(reason=reason))

    data = {"a": [1, 2], "b": [["x", "y"], ["x", "z"]]}

    b_expr = nw.col("b").cast(nw.List(nw.String()))
    left = (
        nw.from_native(constructor(data)).with_columns(b_expr).sort("a", descending=False)
    )
    right = (
        nw.from_native(constructor(data)).with_columns(b_expr).sort("a", descending=True)
    )

    context = (
        _assertion_error('value mismatch for column "a"')
        if check_row_order and left.implementation in GUARANTEES_ROW_ORDER
        else does_not_raise()
    )

    with context:
        assert_frame_equal(left, right, check_row_order=check_row_order)


def test_check_row_order_nested_only(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    if "dask" in str(constructor):
        reason = "Unsupported List type"
        request.applymarker(pytest.mark.xfail(reason=reason))

    data = {"b": [["x", "y"], ["x", "z"]]}

    b_expr = nw.col("b").cast(nw.List(nw.String()))
    left = nw.from_native(constructor(data)).select(b_expr)

    msg = "`check_row_order=False` is not supported (yet) with only nested data type."
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        assert_frame_equal(left, left, check_row_order=False)


def test_values_mismatch(constructor: Constructor) -> None: ...


def test_self_equal(constructor: Constructor, data: Data) -> None:
    """Test that a dataframe is equal to itself, including nested dtypes with nulls.

    We are dropping columns which type is unsupported by _some_ backend.
    """
    cols_to_drop = ("categorical", "enum", "duration", "struct", "time")

    if "pandas" in str(constructor) and PANDAS_VERSION < (2, 2):  # pragma: no cover
        reason = "Pandas too old for nested dtypes"
        pytest.skip(reason=reason)

    _data = {k: v for k, v in data.items() if k not in cols_to_drop}
    df = nw.from_native(constructor(_data))
    assert_frame_equal(df, df)
