from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import narwhals.stable.v1 as nw
from tests.utils import NUMPY_VERSION
from tests.utils import Constructor

if TYPE_CHECKING:
    from narwhals.typing import Frame


def test_all_vs_all(constructor: Constructor) -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6]}
    df: Frame = nw.from_native(constructor(data))
    with pytest.raises(ValueError, match="Multi-output"):
        df.select(nw.all() + nw.all())


def test_invalid() -> None:
    data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
    df: Frame = nw.from_native(pa.table({"a": [1, 2], "b": [3, 4]}))
    with pytest.raises(ValueError, match="Multi-output"):
        df.select(nw.all() + nw.all())
    df = nw.from_native(pd.DataFrame(data))
    with pytest.raises(ValueError, match="Multi-output"):
        df.select(nw.all() + nw.all())
    with pytest.raises(TypeError, match="Perhaps you"):
        df.select([pl.col("a")])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="Expected Narwhals dtype"):
        df.select([nw.col("a").cast(pl.Int64)])  # type: ignore[arg-type]


def test_native_vs_non_native() -> None:
    s_pd = pd.Series([1, 2, 3])
    df_pd = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(TypeError, match="Perhaps you forgot"):
        nw.from_native(df_pd).filter(s_pd > 1)  # type: ignore[arg-type]
    s_pl = pl.Series([1, 2, 3])
    df_pl = pl.DataFrame({"a": [2, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(TypeError, match="Perhaps you\n- forgot"):
        nw.from_native(df_pl).filter(s_pl > 1)


def test_validate_laziness() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    with pytest.raises(
        TypeError,
        match=("The items to concatenate should either all be eager, or all lazy"),
    ):
        nw.concat([nw.from_native(df, eager_only=True), nw.from_native(df).lazy()])


@pytest.mark.slow
@pytest.mark.skipif(NUMPY_VERSION < (1, 26, 4), reason="too old")
def test_memmap() -> None:
    pytest.importorskip("sklearn")
    # the headache this caused me...
    from sklearn.utils import check_X_y
    from sklearn.utils._testing import create_memmap_backed_data

    x_any = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    y_any = create_memmap_backed_data(x_any["b"])

    x_any, y_any = create_memmap_backed_data([x_any, y_any])

    x = nw.from_native(x_any)
    x = x.with_columns(y=nw.from_native(y_any, series_only=True))

    # check this doesn't raise
    check_X_y(nw.to_native(x), nw.to_native(x["y"]))
