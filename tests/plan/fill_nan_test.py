from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
import narwhals._plan as nwp
import narwhals._plan.selectors as ncs
from tests.plan.utils import assert_equal_data, assert_equal_series, dataframe, series

if TYPE_CHECKING:
    from narwhals._plan.typing import OneOrIterable
    from tests.conftest import Data

pytest.importorskip("pyarrow")


@pytest.fixture(scope="module")
def data() -> Data:
    return {"int": [-1, 1, None]}


@pytest.mark.parametrize(
    ("exprs", "expected"),
    [
        (
            [nwp.col("no_nan").fill_nan(None), nwp.col("float_nan").fill_nan(None)],
            [None, 1.0, None],
        ),
        (
            [nwp.col("no_nan").fill_nan(3.0), nwp.col("float_nan").fill_nan(3.0)],
            [3.0, 1.0, None],
        ),
        (nwp.all().fill_nan(None), [None, 1.0, None]),
        (nwp.all().fill_nan(3.0), [3.0, 1.0, None]),
        (
            ncs.numeric().as_expr().fill_nan(nwp.lit(series([55.5, -100, -200]))),
            [55.5, 1.0, None],
        ),
        (
            [
                nwp.col("no_nan"),
                nwp.col("float_nan").fill_nan(nwp.col("no_nan").max() * 6),
            ],
            [6.0, 1.0, None],
        ),
    ],
)
def test_fill_nan(
    data: Data, exprs: OneOrIterable[nwp.Expr], expected: list[Any]
) -> None:
    base = nwp.col("int")
    df = dataframe(data).select(
        base.cast(nw.Float64).alias("no_nan"), (base**0.5).alias("float_nan")
    )
    result = df.select(exprs)
    assert_equal_data(result, {"no_nan": [-1.0, 1.0, None], "float_nan": expected})
    assert result.get_column("float_nan").null_count() == expected.count(None)


def test_fill_nan_series(data: Data) -> None:
    ser = dataframe(data).select(float_nan=nwp.col("int") ** 0.5).get_column("float_nan")
    result = ser.fill_nan(999)
    assert_equal_series(result, [999.0, 1.0, None], "float_nan")
    result = ser.fill_nan(series([1.23, None, None]))
    assert_equal_series(result, [1.23, 1.0, None], "float_nan")
