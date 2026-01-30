from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    import narwhals as nw


TPCHBackend: TypeAlias = Literal[
    "polars[lazy]", "pyarrow", "pandas[pyarrow]", "dask", "duckdb", "sqlframe"
]
QueryID: TypeAlias = Literal[
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "q7",
    "q8",
    "q9",
    "q10",
    "q11",
    "q12",
    "q13",
    "q14",
    "q15",
    "q16",
    "q17",
    "q18",
    "q19",
    "q20",
    "q21",
    "q22",
]
ScaleFactor: TypeAlias = Literal[
    "0.014", "0.052", "0.1", "0.25", "0.51", "1.0", "10.0", "30.0"
]
"""Values for `scale_factor` that are known to produce correct results.

These three are blessed by [TPC-H v3.0.1 (Page 79)]:

    "1.0", "10.0", "30.0"

These five are *not*, but represent a [benchmark runtime] between 13-72 seconds:

    "0.014", "0.052", "0.1", "0.25", "0.51"

Warning:
    Running the higher values can **easily** crash when combined with [`pytest-xdist`].
    We are effectively running `scale_factor * 6` when all backends are selected.

[TPC-H v3.0.1 (Page 79)]: https://www.tpc.org/TPC_Documents_Current_Versions/pdf/TPC-H_v3.0.1.pdf
[benchmark runtime]: https://github.com/narwhals-dev/narwhals/pull/3421#discussion_r2743356336
[`pytest-xdist`]: https://pytest-xdist.readthedocs.io/en/stable/
"""

Artifact: TypeAlias = Literal["database", "answers"]


class QueryModule(Protocol):
    def query(
        self, *args: nw.LazyFrame[Any], **kwds: nw.LazyFrame[Any]
    ) -> nw.LazyFrame[Any]: ...
