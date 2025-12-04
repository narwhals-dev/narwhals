from __future__ import annotations

import pytest

import narwhals as nw
from tests.utils import POLARS_VERSION, Constructor

pytest.importorskip("pyarrow")

data = {"a": [1, 2, 3], "b": ["dogs", "cats", None], "c": ["play", "swim", "walk"]}


def test_dryrun(constructor: Constructor, *, request: pytest.FixtureRequest) -> None:
    if "polars" in str(constructor) and POLARS_VERSION < (1, 0, 0):
        # nth only available after 1.0
        request.applymarker(pytest.mark.xfail)

    # FIXME: for now only polars>1.0 works.
    if "polars" not in str(constructor):
        request.applymarker(pytest.mark.xfail)

    df = nw.from_native(constructor(data))
    result = df.select(nw.concat_list([nw.col("a"), nw.col("b"), nw.col("c")]).alias("s"))

    # FIXME: assert instead of print
    print(result)
