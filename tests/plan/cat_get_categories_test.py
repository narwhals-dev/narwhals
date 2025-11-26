from __future__ import annotations

import pytest

import narwhals as nw
import narwhals._plan as nwp  # noqa: F401
import narwhals._plan.selectors as ncs
from narwhals._utils import Implementation
from tests.plan.utils import assert_equal_data, dataframe
from tests.utils import PYARROW_VERSION

pytest.importorskip("pyarrow")


@pytest.mark.parametrize(
    ("values", "expected"),
    [(["one", "two", "two"], ["one", "two"]), (["A", "B", None, "D"], ["A", "B", "D"])],
    ids=["full", "nulls"],
)
def test_get_categories(
    values: list[str], expected: list[str], request: pytest.FixtureRequest
) -> None:
    data = {"a": values}
    df = dataframe(data)
    request.applymarker(
        pytest.mark.xfail(
            (df.implementation is Implementation.PYARROW and PYARROW_VERSION < (15,)),
            reason="Unsupported cast from string to dictionary using function cast_dictionary",
        )
    )
    df = dataframe(data)
    df = dataframe(data).select(ncs.first().cast(nw.Categorical))
    result = df.select(ncs.first().cat.get_categories())
    assert_equal_data(result, {"a": expected})
