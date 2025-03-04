from __future__ import annotations

import sys
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from importlib import import_module
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Generator
from typing import TypeVar

from narwhals.dataframe import DataFrame
from narwhals.dataframe import LazyFrame
from narwhals.series import Series
from narwhals.utils import Implementation
from narwhals.utils import Version
from narwhals.utils import parse_version

if TYPE_CHECKING:
    from types import ModuleType

    T = TypeVar("T")


BACKENDS = []


@dataclass
class Adaptation:
    narwhals: type
    native: str | type
    adapter: str | type
    level: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    version: Version | None = None

    @property
    def imported_adapter(self) -> type:
        obj = dynamic_import(self.adapter)
        if not isinstance(obj, type):
            msg = f"Attempted to import {self.adapter!r}, expected an instance of type but got {obj}"
            raise TypeError(msg)
        return obj

    @property
    def imported_native(self) -> type:
        obj = dynamic_import(self.adapter)
        if not isinstance(obj, type):
            msg = f"Attempted to import {self.adapter!r}, expected an instance of type but got {obj}"
            raise TypeError(msg)
        return obj


@dataclass
class Backend:
    requires: list[tuple[str, str | Callable[[], str], tuple[int, ...]]]
    adaptations: list[Adaptation]
    implementation: Implementation | None = None

    def __post_init__(self) -> None:
        adaptations = []
        for adapt in self.adaptations:
            if adapt.version in Version:
                adaptations.append(adapt)
            elif adapt.version is None:
                adaptations.extend(replace(adapt, version=v) for v in Version)
            else:
                msg = "Adaptation.version must be {Version!r} or None, got {adapt.version!r}"
                raise TypeError(msg)
        self.adaptations = adaptations

    def get_adapter(
        self, cls: type, version: Version = Version.MAIN
    ) -> Adaptation | None:
        module_name, *_ = cls.__module__.split(".", maxsplit=1)
        for adapt in self.adaptations:
            if adapt.version != version:
                continue

            if isinstance(adapt.native, type) and cls is adapt.native:
                return adapt

            elif isinstance(adapt.native, str):
                adapt_module_name, *_, adapt_cls_name = adapt.native.split(".")
                if (
                    (adapt_module_name in sys.modules)  # base-module is imported
                    and (module_name == adapt_module_name)  # roots match
                    and (cls.__name__ == adapt_cls_name)  # tips match
                    and (cls is dynamic_import(adapt.native))  # types are identical
                ):
                    return adapt
        return None

    def validate_backend_version(self) -> None:
        for module_name, version_getter, min_version in self.requires:
            # TODO(camriddell): this logic may be better suited for a Version namedtuple or dataclass
            if callable(version_getter):
                version_str = version_getter()
            elif isinstance(version_getter, str):
                version_str = dynamic_import(version_getter)
            else:
                msg = "version_getter {version_getter!r} must be a string or callable, got {type(version_getter)}"
                raise TypeError(msg)

            installed_version = parse_version(version_str)
            if installed_version < min_version:
                msg = f"{module_name} must be updated to at least {min_version}, got {installed_version}"
                raise ValueError(msg)

    def version(self) -> tuple[int, ...]:
        version_getter = self.requires[0][1]
        # TODO(camriddell): this logic may be better suited for a Version namedtuple or dataclass
        if callable(version_getter):
            version_str = version_getter()
        elif isinstance(version_getter, str):
            version_str = dynamic_import(version_getter)
        else:
            msg = "version_getter {version_getter!r} must be a string or callable, got {type(version_getter)}"
            raise TypeError(msg)
        return parse_version(version_str)

    def native_namespace(self) -> ModuleType:
        return import_module(self.requires[0][0])

    def get_native_namespace(self) -> ModuleType | None:
        return sys.modules.get(self.requires[0][0], None)


def register_backends(*backends: Backend) -> None:
    for b in backends:
        BACKENDS.append(b)  # noqa: PERF402


def traverse_rsplits(text: str, sep: str = " ") -> Generator[tuple[str, list[str]]]:
    sep_count = text.count(sep)
    if sep_count == 0:
        yield (text, [])

    for i in range(1, sep_count + 1):
        base, *remaining = text.rsplit(sep, maxsplit=i)
        yield base, remaining


def dynamic_import(dotted_path: str | type, /) -> Any:
    if isinstance(dotted_path, type):
        return dotted_path
    for base, attributes in traverse_rsplits(dotted_path, sep="."):
        if not attributes:
            continue
        try:
            module = import_module(base)
        except ImportError:
            pass
        else:
            obj = module
            for attr in attributes:
                obj = getattr(obj, attr)
            return obj
    msg = "Could not import {dotted_path!r}"
    raise ImportError(msg)


register_backends(
    Backend(
        requires=[
            ("pandas", "pandas.__version__", (0, 25, 3)),
        ],
        adaptations=[
            Adaptation(
                DataFrame,
                "pandas.DataFrame",
                "narwhals._pandas_like.dataframe.PandasLikeDataFrame",
                level="full",
                kwargs={"validate_column_names": True},
            ),
            Adaptation(
                Series,
                "pandas.Series",
                "narwhals._pandas_like.dataframe.PandasLikeSeries",
                level="full",
            ),
        ],
        implementation=Implementation.PANDAS,
    ),
    Backend(
        requires=[
            ("polars", "polars.__version__", (0, 20, 3)),
        ],
        adaptations=[
            Adaptation(
                LazyFrame,
                "polars.LazyFrame",
                "narwhals._polars.dataframe.PolarsLazyFrame",
                level="full",
            ),
            Adaptation(
                DataFrame,
                "polars.DataFrame",
                "narwhals._polars.dataframe.PolarsDataFrame",
                level="full",
            ),
            Adaptation(
                Series,
                "polars.Series",
                "narwhals._polars.series.PolarsSeries",
                level="full",
            ),
        ],
    ),
    Backend(
        requires=[("modin.pandas", "modin.__version__", (0, 25, 3))],
        adaptations=[
            Adaptation(
                DataFrame,
                "modin.pandas.DataFrame",
                "narwhals._pandas_like.dataframe.PandasLikeDataFrame",
                level="full",
                kwargs={"validate_column_names": True},
            ),
            Adaptation(
                Series,
                "modin.pandas.Series",
                "narwhals._pandas_like.dataframe.PandasLikeSeries",
                level="full",
            ),
        ],
        implementation=Implementation.MODIN,
    ),
    Backend(
        requires=[
            ("cudf", "cudf.__version__", (24, 10)),
        ],
        adaptations=[
            Adaptation(
                DataFrame,
                "cudf.DataFrame",
                "narwhals._pandas_like.dataframe.PandasLikeDataFrame",
                level="full",
                kwargs={"validate_column_names": True},
            ),
            Adaptation(
                Series,
                "cudf.Series",
                "narwhals._pandas_like.dataframe.PandasLikeSeries",
                level="full",
            ),
        ],
        implementation=Implementation.CUDF,
    ),
    Backend(
        requires=[
            ("pyarrow", "pyarrow.__version__", (11,)),
        ],
        adaptations=[
            Adaptation(
                DataFrame,
                "pyarrow.Table",
                "narwhals._arrow.dataframe.ArrowDataFrame",
                level="full",
                kwargs={"validate_column_names": True},
            ),
            Adaptation(
                Series,
                "pyarrow.ChunkedArray",
                "narwhals._arrow.series.ArrowSeries",
                level="full",
                kwargs={"name": ""},
            ),
        ],
    ),
    Backend(
        requires=[("pyspark.sql", "pyspark.__version__", (3, 5))],
        adaptations=[
            Adaptation(
                LazyFrame,
                "pyspark.sql.DataFrame",
                "narwhals._spark.dataframe.SparkLikeLazyFrame",
                level="full",
                kwargs={"validate_column_names": True},
            ),
            Adaptation(
                Series,
                "pyspark.sql.Series",
                "narwhals._arrow.dataframe.ArrowSeries",
                level="full",
            ),
        ],
        implementation=Implementation.PYSPARK,
    ),
    Backend(
        requires=[
            ("dask.dataframe", "dask.__version__", (2024, 8)),
            ("dask_expr", "dask_expr.__version__", (0,)),
        ],
        adaptations=[
            Adaptation(
                LazyFrame,
                "dask.dataframe.DataFrame",
                "narwhals._dask.dataframe.DaskLazyFrame",
                level="full",
                kwargs={"validate_column_names": True},
            ),
            Adaptation(
                LazyFrame,
                "dask_expr.DataFrame",
                "narwhals._dask.dataframe.DaskLazyFrame",
                level="full",
                kwargs={"validate_column_names": True},
            ),
        ],
    ),
    Backend(
        requires=[("duckdb", "duckdb.__version__", (1,))],
        adaptations=[
            Adaptation(
                LazyFrame,
                "duckdb.DuckDBPyRelation",
                "narwhals._duckdb.dataframe.DuckDBLazyFrame",
                level="full",
                kwargs={"validate_column_names": True},
                version=Version.MAIN,
            ),
            Adaptation(
                DataFrame,
                "duckdb.DuckDBPyRelation",
                "narwhals._duckdb.dataframe.DuckDBLazyFrame",
                level="interchange",
                version=Version.V1,
                kwargs={"validate_column_names": True},
            ),
        ],
    ),
    Backend(
        requires=[
            ("ibis", "ibis.__version__", (6,)),
        ],
        adaptations=[
            Adaptation(
                LazyFrame,
                "ibis.expr.types.Table",
                "narwhals._ibis.dataframe.IbisLazyFrame",
                level="full",
                kwargs={"validate_column_names": True},
            ),
        ],
    ),
    Backend(
        requires=[
            ("sqlframe", "sqlframe._version.__version__", (3, 14, 2)),
        ],
        adaptations=[
            Adaptation(
                LazyFrame,
                "sqlframe.base.dataframe.BaseDataFrame",
                "narwhals._spark.dataframe.SparkLikeLazyFrame",
                level="full",
                kwargs={"validate_column_names": True},
            ),
        ],
        implementation=Implementation.SQLFRAME,
    ),
)
