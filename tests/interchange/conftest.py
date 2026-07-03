from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import pytest

import narwhals as nw

if TYPE_CHECKING:
    from collections.abc import Callable

    import duckdb
    import ibis

    from narwhals._typing import EagerAllowed

Interchange: TypeAlias = "duckdb.DuckDBPyRelation | ibis.Table"
MainInstances: TypeAlias = tuple[nw.DataFrame[Any], nw.LazyFrame[Any], nw.Series[Any]]


@pytest.fixture
def main_instances(eager_implementation: EagerAllowed) -> MainInstances:
    df = nw.DataFrame.from_dict({"a": [1, 2, 3]}, backend=eager_implementation)
    return df, df.lazy(), df.get_column("a")


class MockDf:
    def __dataframe__(self) -> None:  # pragma: no cover
        return


@pytest.fixture
def mockdf() -> MockDf:
    return MockDf()


@pytest.fixture
def frame(constructor: Callable[[Any], Interchange]) -> Interchange:
    name = str(constructor)
    if "duckdb" in name or "ibis" in name:
        return constructor({"a": [1, 2, 3], "b": [4, 5, 6]})
    pytest.skip("non-interchange frames are checked in other tests")
