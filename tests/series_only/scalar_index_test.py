from __future__ import annotations

import narwhals.stable.v1 as nw
from tests.utils import assert_equal_data


def test_scalar_index() -> None:
    np = nw.dependencies.get_numpy()
    pd = nw.dependencies.get_pandas()
    s = pd.Series([0, 1, 2])
    snw = nw.from_native(s, series_only=True)
    assert_equal_data(snw[snw.min()], np.int64(0))
    assert_equal_data(snw[0], np.int64(0))
