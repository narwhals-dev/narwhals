from __future__ import annotations

import pytest

import narwhals._plan as nwp
from tests.plan.utils import assert_equal_data, dataframe
from tests.utils import PYARROW_VERSION


@pytest.mark.xfail(
    PYARROW_VERSION < (21,), reason="TODO: `str.zfill` port", raises=NotImplementedError
)
def test_str_zfill() -> None:  # pragma: no cover
    data = {"a": ["-1", "+1", "1", "12", "123", "99999", "+9999", None]}
    expected = {"a": ["-01", "+01", "001", "012", "123", "99999", "+9999", None]}
    result = dataframe(data).select(nwp.col("a").str.zfill(3))
    assert_equal_data(result, expected)
