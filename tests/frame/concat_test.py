from __future__ import annotations

import datetime as dt
import re
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
from narwhals._utils import Implementation
from narwhals.exceptions import InvalidOperationError, NarwhalsError
from narwhals.schema import Schema
from tests.utils import POLARS_VERSION, Constructor, ConstructorEager, assert_equal_data

if TYPE_CHECKING:
    from collections.abc import Iterator

    from narwhals.dtypes import DType
    from narwhals.typing import IntoSchema

if TYPE_CHECKING:
    from narwhals.typing import LazyFrameT


def _cast(frame: LazyFrameT, schema: IntoSchema) -> LazyFrameT:
    return frame.select(nw.col(name).cast(dtype) for name, dtype in schema.items())


def test_concat_horizontal(constructor_eager: ConstructorEager) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df_left = nw.from_native(constructor_eager(data), eager_only=True)

    data_right = {"c": [6, 12, -1], "d": [0, -4, 2]}
    df_right = nw.from_native(constructor_eager(data_right), eager_only=True)

    result = nw.concat([df_left, df_right], how="horizontal")
    expected = {
        "a": [1, 3, 2],
        "b": [4, 4, 6],
        "z": [7.0, 8.0, 9.0],
        "c": [6, 12, -1],
        "d": [0, -4, 2],
    }
    assert_equal_data(result, expected)

    with pytest.raises(ValueError, match="No items"):
        nw.concat([])
    pattern = re.compile(r"horizontal.+not supported.+lazyframe", re.IGNORECASE)
    with pytest.raises(InvalidOperationError, match=pattern):
        nw.concat([df_left.lazy()], how="horizontal")


def test_concat_vertical(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df_left = (
        nw.from_native(constructor(data)).lazy().rename({"a": "c", "b": "d"}).drop("z")
    )

    data_right = {"c": [6, 12, -1], "d": [0, -4, 2]}
    df_right = nw.from_native(constructor(data_right)).lazy()

    result = nw.concat([df_left, df_right], how="vertical")
    expected = {"c": [1, 3, 2, 6, 12, -1], "d": [4, 4, 6, 0, -4, 2]}
    assert_equal_data(result, expected)

    with pytest.raises(ValueError, match="No items"):
        nw.concat([], how="vertical")

    with pytest.raises(
        (Exception, TypeError),
        match=r"unable to vstack|inputs should all have the same schema",
    ):
        nw.concat([df_left, df_right.rename({"d": "i"})], how="vertical").collect()
    with pytest.raises(
        (Exception, TypeError),
        match=r"unable to vstack|unable to append|inputs should all have the same schema",
    ):
        nw.concat([df_left, df_left.select("d")], how="vertical").collect()


def test_concat_diagonal(constructor: Constructor) -> None:
    data_1 = {"a": [1, 3], "b": [4, 6]}
    data_2 = {"a": [100, 200], "z": ["x", "y"]}
    expected = {
        "a": [1, 3, 100, 200],
        "b": [4, 6, None, None],
        "z": [None, None, "x", "y"],
    }

    df_1 = nw.from_native(constructor(data_1)).lazy()
    df_2 = nw.from_native(constructor(data_2)).lazy()

    result = nw.concat([df_1, df_2], how="diagonal")

    assert_equal_data(result, expected)

    with pytest.raises(ValueError, match="No items"):
        nw.concat([], how="diagonal")


def _from_natives(
    constructor: Constructor, *sources: dict[str, list[Any]]
) -> Iterator[nw.LazyFrame[Any]]:
    yield from (nw.from_native(constructor(data)).lazy() for data in sources)


def test_concat_diagonal_bigger(constructor: Constructor) -> None:
    # NOTE: `ibis.union` doesn't guarantee the order of outputs
    # https://github.com/narwhals-dev/narwhals/pull/3404#discussion_r2694556781
    data_1 = {"idx": [1, 2], "a": [1, 2], "b": [3, 4]}
    data_2 = {"a": [5, 6], "c": [7, 8], "idx": [3, 4]}
    data_3 = {"b": [9, 10], "idx": [5, 6], "c": [11, 12]}
    expected = {
        "idx": [1, 2, 3, 4, 5, 6],
        "a": [1, 2, 5, 6, None, None],
        "b": [3, 4, None, None, 9, 10],
        "c": [None, None, 7, 8, 11, 12],
    }
    dfs = _from_natives(constructor, data_1, data_2, data_3)
    result = nw.concat(dfs, how="diagonal").sort("idx")
    assert_equal_data(result, expected)


def test_concat_diagonal_invalid(
    constructor: Constructor, request: pytest.FixtureRequest
) -> None:
    data_1 = {"a": [1, 3], "b": [4, 6]}
    data_2 = {
        "a": [dt.datetime(2000, 1, 1), dt.datetime(2000, 1, 2)],
        "b": [4, 6],
        "z": ["x", "y"],
    }
    df_1 = nw.from_native(constructor(data_1)).lazy()
    bad_schema = nw.from_native(constructor(data_2)).lazy()
    impl = df_1.implementation
    request.applymarker(
        pytest.mark.xfail(
            impl not in {Implementation.IBIS, Implementation.POLARS},
            reason=f"{impl!r} does not validate schemas for `concat(how='diagonal')",
        )
    )
    context: Any
    if impl.is_polars() and POLARS_VERSION < (1,):  # pragma: no cover
        context = pytest.raises(
            NarwhalsError,
            match=re.compile(r"(int.+datetime)|(datetime.+int)", re.IGNORECASE),
        )
    else:
        context = pytest.raises((InvalidOperationError, TypeError), match=r"same schema")
    with context:
        nw.concat([df_1, bad_schema], how="diagonal").collect().to_dict(as_series=False)


@pytest.mark.parametrize(
    ("ldata", "lschema", "rdata", "rschema", "expected_data", "expected_schema"),
    [
        (
            {"a": [1, 2, 3], "b": [True, False, None]},
            {"a": nw.Int8(), "b": nw.Boolean()},
            {"a": [43, 2, 3], "b": [32, 1, None]},
            {"a": nw.Int16(), "b": nw.Int64()},
            {"a": [1, 2, 3, 43, 2, 3], "b": [1, 0, None, 32, 1, None]},
            {"a": nw.Int16(), "b": nw.Int64()},
        ),
        (
            {"a": [1, 2], "b": [2, 1]},
            {"a": nw.Int32(), "b": nw.Int32()},
            {"a": [1.0, 0.2], "b": [None, 0.1]},
            {"a": nw.Float32(), "b": nw.Float32()},
            {"a": [1.0, 2.0, 1.0, 0.2], "b": [2.0, 1.0, None, 0.1]},
            {"a": nw.Float64(), "b": nw.Float64()},
        ),
    ],
    ids=["nullable-integer", "nullable-float"],
)
def test_concat_vertically_relaxed(
    constructor: Constructor,
    ldata: dict[str, Any],
    lschema: dict[str, DType],
    rdata: dict[str, Any],
    rschema: dict[str, DType],
    expected_data: dict[str, Any],
    expected_schema: dict[str, DType],
    request: pytest.FixtureRequest,
) -> None:
    # Adapted from https://github.com/pola-rs/polars/blob/b0fdbd34d430d934bda9a4ca3f75e136223bd95b/py-polars/tests/unit/functions/test_concat.py#L64
    is_nullable_int = request.node.callspec.id.endswith("nullable-integer")
    if is_nullable_int and any(
        x in str(constructor)
        for x in ("dask", "pandas_constructor", "modin_constructor", "cudf")
    ):
        reason = "Cannot convert non-finite values (NA or inf)"
        request.applymarker(pytest.mark.xfail(reason=reason))
    left = nw.from_native(constructor(ldata)).lazy().pipe(_cast, lschema)
    right = nw.from_native(constructor(rdata)).lazy().pipe(_cast, rschema)
    result = nw.concat([left, right], how="vertical_relaxed")

    assert result.collect_schema() == Schema(expected_schema)
    assert_equal_data(result.collect(), expected_data)

    result = nw.concat([right, left], how="vertical_relaxed")
    assert result.collect_schema() == Schema(expected_schema)


@pytest.mark.parametrize(
    ("schema1", "schema2", "schema3", "expected_schema"),
    [
        (
            {"a": nw.Int32(), "c": nw.Int64()},
            {"a": nw.Float64(), "b": nw.Float32()},
            {"b": nw.Int32(), "c": nw.Int32()},
            {"a": nw.Float64(), "c": nw.Int64(), "b": nw.Float64()},
        ),
        (
            {"a": nw.Float32(), "c": nw.Float32()},
            {"a": nw.Float64(), "b": nw.Float32()},
            {"b": nw.Float32(), "c": nw.Float32()},
            {"a": nw.Float64(), "c": nw.Float32(), "b": nw.Float32()},
        ),
    ],
    ids=["nullable-integer", "nullable-float"],
)
def test_concat_diagonal_relaxed(
    constructor: Constructor,
    schema1: dict[str, DType],
    schema2: dict[str, DType],
    schema3: dict[str, DType],
    expected_schema: dict[str, DType],
    request: pytest.FixtureRequest,
) -> None:
    # Adapted from https://github.com/pola-rs/polars/blob/b0fdbd34d430d934bda9a4ca3f75e136223bd95b/py-polars/tests/unit/functions/test_concat.py#L265C1-L288C41
    is_nullable_int = request.node.callspec.id.endswith("nullable-integer")
    if is_nullable_int and any(
        x in str(constructor)
        for x in ("dask", "pandas_constructor", "modin_constructor", "cudf")
    ):
        reason = "Cannot convert non-finite values (NA or inf)"
        request.applymarker(pytest.mark.xfail(reason=reason))

    base_schema = {"idx": nw.Int32()}

    data1 = {"idx": [0, 1], "a": [1, 2], "c": [10, 20]}
    df1 = nw.from_native(constructor(data1)).lazy().pipe(_cast, base_schema | schema1)

    data2 = {"a": [3.5, 4.5], "b": [30.1, 40.2], "idx": [2, 3]}
    df2 = nw.from_native(constructor(data2)).lazy().pipe(_cast, base_schema | schema2)

    data3 = {"b": [5, 6], "idx": [4, 5], "c": [50, 60]}
    df3 = nw.from_native(constructor(data3)).lazy().pipe(_cast, base_schema | schema3)

    result = nw.concat([df1, df2, df3], how="diagonal_relaxed").sort("idx")
    out_schema = result.collect_schema()
    assert out_schema == Schema(base_schema | expected_schema)

    expected_data = {
        "idx": [0, 1, 2, 3, 4, 5],
        "a": [1.0, 2.0, 3.5, 4.5, None, None],
        "c": [10, 20, None, None, 50, 60],
        "b": [None, None, 30.1, 40.2, 5.0, 6.0],
    }
    assert_equal_data(result.collect(), expected_data)
