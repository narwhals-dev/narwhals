from __future__ import annotations

import narwhals.stable.v1 as nw
from tests.utils import Constructor
from tests.utils import assert_equal_data


def test_collect_kwargs(constructor: Constructor) -> None:
    data = {"a": [1, 2], "b": [3, 4]}
    df = nw.from_native(constructor(data))

    result = (
        df.lazy()
        .select(nw.all().sum())
        .collect(
            polars_kwargs={"no_optimization": True},
            dask_kwargs={"optimize_graph": False},
        )
    )

    expected = {"a": [3], "b": [7]}
    assert_equal_data(result, expected)
