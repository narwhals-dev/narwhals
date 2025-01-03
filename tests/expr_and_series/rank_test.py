from __future__ import annotations

from contextlib import nullcontext as does_not_raise
from typing import Literal

import pytest

import narwhals.stable.v1 as nw
from tests.utils import PANDAS_VERSION
from tests.utils import Constructor
from tests.utils import ConstructorEager
from tests.utils import assert_equal_data

rank_methods = ["average", "min", "max", "dense", "ordinal"]

data_int = {"a": [3, 6, 1, 1, None, 6], "b": [1, 1, 2, 1, 2, 2]}
data_float = {"a": [3.1, 6.1, 1.5, 1.5, None, 6.1], "b": [1, 1, 2, 1, 2, 2]}

expected = {
    "average": [3.0, 4.5, 1.5, 1.5, None, 4.5],
    "min": [3, 4, 1, 1, None, 4],
    "max": [3, 5, 2, 2, None, 5],
    "dense": [2, 3, 1, 1, None, 3],
    "ordinal": [3, 4, 1, 2, None, 5],
}

expected_over = {
    "average": [2.0, 3.0, 1.0, 1.0, None, 2.0],
    "min": [2, 3, 1, 1, None, 2],
    "max": [2, 3, 1, 1, None, 2],
    "dense": [2, 3, 1, 1, None, 2],
    "ordinal": [2, 3, 1, 1, None, 2],
}


@pytest.mark.parametrize("method", rank_methods)
@pytest.mark.parametrize("data", [data_int, data_float])
def test_rank_expr(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    method: Literal["average", "min", "max", "dense", "ordinal"],
    data: dict[str, list[float]],
) -> None:
    if "dask" in str(constructor):
        request.applymarker(pytest.mark.xfail)

    if (
        "pandas_pyarrow" in str(constructor)
        and PANDAS_VERSION < (2, 1)
        and isinstance(data["a"][0], int)
    ):
        request.applymarker(pytest.mark.xfail)

    context = (
        pytest.raises(
            ValueError,
            match=r"`rank` with `method='average' is not supported for pyarrow backend.",
        )
        if "pyarrow_table" in str(constructor) and method == "average"
        else does_not_raise()
    )

    with context:
        df = nw.from_native(constructor(data))

        result = df.select(nw.col("a").rank(method=method))
        expected_data = {"a": expected[method]}
        assert_equal_data(result, expected_data)


@pytest.mark.parametrize("method", rank_methods)
@pytest.mark.parametrize("data", [data_int, data_float])
def test_rank_series(
    request: pytest.FixtureRequest,
    constructor_eager: ConstructorEager,
    method: Literal["average", "min", "max", "dense", "ordinal"],
    data: dict[str, list[float]],
) -> None:
    if (
        "pandas_pyarrow" in str(constructor_eager)
        and PANDAS_VERSION < (2, 1)
        and isinstance(data["a"][0], int)
    ):
        request.applymarker(pytest.mark.xfail)

    context = (
        pytest.raises(
            ValueError,
            match=r"`rank` with `method='average' is not supported for pyarrow backend.",
        )
        if "pyarrow_table" in str(constructor_eager) and method == "average"
        else does_not_raise()
    )

    with context:
        df = nw.from_native(constructor_eager(data), eager_only=True)

        result = {"a": df["a"].rank(method=method)}
        expected_data = {"a": expected[method]}
        assert_equal_data(result, expected_data)


@pytest.mark.parametrize("method", rank_methods)
def test_rank_expr_in_over_context(
    request: pytest.FixtureRequest,
    constructor: Constructor,
    method: Literal["average", "min", "max", "dense", "ordinal"],
) -> None:
    if any(x in str(constructor) for x in ("pyarrow_table", "dask")):
        # Pyarrow raises:
        # > pyarrow.lib.ArrowKeyError: No function registered with name: hash_rank
        # We can handle that to provide a better error message.
        request.applymarker(pytest.mark.xfail)

    if "pandas_pyarrow" in str(constructor) and PANDAS_VERSION < (2, 1):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data_float))

    result = df.select(nw.col("a").rank(method=method).over("b"))
    expected_data = {"a": expected_over[method]}
    assert_equal_data(result, expected_data)


def test_invalid_method_raise(constructor: Constructor) -> None:
    method = "invalid_method_name"
    df = nw.from_native(constructor(data_float))

    msg = (
        "Ranking method must be one of {'average', 'min', 'max', 'dense', 'ordinal'}. "
        f"Found '{method}'"
    )

    with pytest.raises(ValueError, match=msg):
        df.select(nw.col("a").rank(method=method))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=msg):
        df.lazy().collect()["a"].rank(method=method)  # type: ignore[arg-type]
