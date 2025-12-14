from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
import narwhals._plan as nwp
from narwhals._plan import selectors as ncs
from tests.plan.utils import assert_equal_data, dataframe, re_compile

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pyarrow as pa

    from narwhals._plan.compliant.typing import (
        SeriesAny as CompliantSeriesAny,
        SeriesT as CompliantSeriesT,
    )
    from narwhals.typing import _1DArray, _NumpyScalar
    from tests.conftest import Data


pytest.importorskip("numpy")
import numpy as np


@pytest.fixture
def data() -> Data:
    return {
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [9, 2, 4],
        "d": [8, 7, 8],
        "e": ["A", "B", "A"],
        "j": [12.1, None, 4.0],
        "k": [42, 10, None],
        "z": [7.0, 8.0, 9.0],
    }


def elementwise_series(s: CompliantSeriesT, /) -> CompliantSeriesT:
    dtype_name = type(s.dtype).__name__.lower()
    repeat_name = (dtype_name,) * (len(s) - 1)
    values = [*repeat_name, "last"]
    return s.from_iterable(values, version=s.version, name="funky")


def elementwise_1d_array(s: CompliantSeriesAny, /) -> _1DArray:
    return s.to_numpy() + 1


def to_numpy(s: CompliantSeriesAny, /) -> _1DArray:
    return s.to_numpy()


def groupwise_1d_array(s: CompliantSeriesAny, /) -> _1DArray:
    result: _1DArray = np.append(s.to_numpy(), [10, 2])
    return result


def aggregation_np_scalar(s: CompliantSeriesAny, /) -> _NumpyScalar:
    result: _NumpyScalar = s.to_numpy().max()
    return result


def aggregation_pa_scalar(s: CompliantSeriesAny) -> pa.Scalar[Any]:
    pytest.importorskip("pyarrow")
    import pyarrow as pa

    result: pa.Scalar[Any] = pa.array(s.to_list())[0]
    return result


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        pytest.param(
            [
                nwp.col("e")
                .alias("...")
                .map_batches(elementwise_series, is_elementwise=True),
                nwp.col("e"),
            ],
            {"funky": ["string", "string", "last"], "e": ["A", "B", "A"]},
            id="is_elementwise-series",
        ),
        pytest.param(
            nwp.col("a", "b", "z").map_batches(to_numpy),
            {"a": [1, 2, 3], "b": [4, 5, 6], "z": [7.0, 8.0, 9.0]},
            id="to-numpy",
        ),
        pytest.param(
            nwp.col("a")
            .map_batches(elementwise_1d_array, nw.Float64, is_elementwise=True)
            .sum(),
            {"a": [9.0]},
            id="is_elementwise-1d-array",
        ),
        pytest.param(
            nwp.col("a").map_batches(elementwise_1d_array, nw.Float64).sum(),
            {"a": [9.0]},
            id="unknown-1d-array",
        ),
        pytest.param(
            ncs.by_index(0, 2, 3)
            .map_batches(groupwise_1d_array, is_elementwise=True)
            .sort(),
            {"a": [1, 2, 2, 3, 10], "c": [2, 2, 4, 9, 10], "d": [2, 7, 8, 8, 10]},
            # NOTE: Maybe this should be rejected because of the length change?
            # It doesn't break broadcasting rules, but uses an optional argument incorrectly
            # and we only know *after* execution
            id="is_elementwise-1d-array-groupwise",
        ),
        pytest.param(
            nwp.col("j", "k")
            .fill_null(15)
            .map_batches(aggregation_np_scalar, returns_scalar=True),
            {"j": [15], "k": [42]},
            id="returns_scalar-np-scalar",
        ),
        pytest.param(
            [
                nwp.col("a").map_batches(
                    lambda _: [1, 2],
                    returns_scalar=True,
                    return_dtype=nw.List(nw.Int64()),
                ),
                nwp.col("b").last(),
            ],
            {"a": [[1, 2]], "b": [6]},
            id="returns_scalar-list",
        ),
    ],
)
def test_map_batches(
    data: Data, expr: nwp.Expr | Sequence[nwp.Expr], expected: Data
) -> None:
    result = dataframe(data).select(expr)
    assert_equal_data(result, expected)


@pytest.mark.parametrize(
    ("udf", "result_type_name"),
    [
        (aggregation_np_scalar, "'numpy.int64'"),
        (aggregation_pa_scalar, ".+pyarrow.+scalar.+"),
        (len, "'int'"),
        (str, "'str'"),
    ],
)
def test_map_batches_invalid(
    data: Data, udf: Callable[[Any], Any], result_type_name: str
) -> None:
    expr = nwp.col("a", "b", "z").map_batches(udf)
    pattern = re_compile(
        rf"map.+ with `returns_scalar=False` must return a Series.+{result_type_name}"
    )
    with pytest.raises(TypeError, match=pattern):
        dataframe(data).select(expr)
