from __future__ import annotations

import narwhals.stable.v1 as nw

np = nw.dependencies.get_numpy()
pd = nw.dependencies.get_pandas()


def test_index() -> None:
    s = pd.Series([0, 1, 2])
    snw = nw.from_native(s, series_only=True)
    assert snw[snw[0]] == np.int64(0)
