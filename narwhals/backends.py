from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from importlib import import_module
from textwrap import indent
from types import ModuleType
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Generator
from typing import Literal
from typing import TypeVar

from narwhals.dataframe import DataFrame
from narwhals.dataframe import LazyFrame
from narwhals.series import Series
from narwhals.utils import Implementation
from narwhals.utils import Version
from narwhals.utils import parse_version

if TYPE_CHECKING:
    T = TypeVar("T")


@dataclass
class Adaptation:
    """Links a Narwhals `interface` to a `native` type through the `adapter` class.

    interface: the narwhals type
    native: the native type (e.g. Polars.LazyFrame, pandas.DataFrame, modin.pandas.DataFrame)
    adapter: the class that implements the API of the interface class using the mechanics of the native class.
    level: The degree of support that Narwhals has for this native class.
    kwargs: Additional kwargs that should be passed to the adapter class when creating an instance.
    version: The version(s) of the Narwhals API this adaptation supports.
    """

    interface: type[LazyFrame[Any] | DataFrame[Any] | Series[Any]]
    native: str | type
    adapter: str | type
    level: Literal["full", "lazy", "interchange"]
    kwargs: dict[str, Any] = field(default_factory=dict)
    version: Version = Version.MAIN

    @property
    def imported_adapter(self) -> type:
        """The object returned by importing `self.adapter`.

        Returns:
            The type specified by `self.adapter`.

        Raises:
            ImportError: If the type specified by the string is unimportable.
        """
        obj = dynamic_import(self.adapter)
        if not isinstance(obj, type):
            msg = f"Attempted to import {self.adapter!r}, expected an instance of type but got {obj}"
            raise TypeError(msg)
        return obj

    @property
    def imported_native(self) -> type:
        """The object returned by importing `self.native`.

        Returns:
            The type specified by `self.native`.

        Raises:
            ImportError: If the type specified by the string is unimportable.
        """
        obj = dynamic_import(self.adapter)
        if not isinstance(obj, type):
            msg = f"Attempted to import {self.adapter!r}, expected an instance of type but got {obj}"
            raise TypeError(msg)
        return obj

    def matches(self, cls: type, version: Version) -> bool:
        """Determines whether this Adapter matches the passed `cls` and `version`.

        Returns:
            True if the native object and version in this Adaptation matches the passed type and version.
            False otherwise.
        """
        module_name, *_ = cls.__module__.split(".", maxsplit=1)

        if version not in self.version:
            return False

        if isinstance(self.native, type) and cls is self.native:
            return True

        elif isinstance(self.native, str):
            adapt_module_name, *_, adapt_cls_name = self.native.split(".")
            if (
                (adapt_module_name in sys.modules)  # base-module is imported
                and (module_name == adapt_module_name)  # roots match
                and (cls.__name__ == adapt_cls_name)  # tips match
                and (cls is dynamic_import(self.native))  # types are identical
            ):
                return True
        return False


@dataclass
class MROAdaptation(Adaptation):
    """An Adaptation that matches the native object to any type in the mro of a passed type.

    Useful if a downstream package has a base-class with multiple subclasses
    that can all be represented with the same Adaptation options.
    """

    def matches(self, cls: type, version: Version) -> bool:
        match_func = super().matches
        return any(match_func(cls=base_cls, version=version) for base_cls in cls.mro())


@dataclass
class Requirement:
    """Represents a package/module requirement with a specified minimum version."""

    module: str
    version_getter: str | Callable[[], str]
    min_version: tuple[int, ...]

    def version(self) -> tuple[int, ...]:
        """Retrieve the version of the imported module.

        Returns:
            The current version of the package/module.
        """
        if callable(self.version_getter):
            version_str = self.version_getter()
        elif isinstance(self.version_getter, str):
            version_str = dynamic_import(self.version_getter)
        else:
            msg = "version_getter {version_getter!r} must be a string or callable, got {type(version_getter)}"
            raise TypeError(msg)

        return parse_version(version_str)

    def is_valid(self) -> bool:
        """Determines whether the imported package meets the requirements specified in this class.

        Returns:
            True if the requirements specified for the module are met. False otherwise
        """
        return self.version() >= self.min_version


@dataclass
class Backend:
    """A collection of metadata that associates a package and its import types to narwhals interface(s).

    requirement: A requirement for the core package that this Backend represents.
    adaptation: a list of Adaptations that link types from the package `requirement` to narwhals interfaces.
    extra_requirements: any additional requirements that should be checked for the the use inside of Narwhals.
    implementation: The narwhals Implementation to be passed to the adapter class (if it requires one).
    """

    requirement: Requirement
    adaptations: list[Adaptation]
    extra_requirements: list[Requirement] = field(default_factory=list)
    implementation: Implementation | None = None

    @property
    def requirements(self) -> Generator[Requirement]:
        """Traverse all requirements in this Backend.

        Yields:
            Each requirement specified across `self.requirement` and `self.extra_requirements`
        """
        yield self.requirement
        yield from self.extra_requirements

    def imported_package(self) -> ModuleType:
        """The imported version of the package specified in `self.requirement`.

        Returns:
            The imported package specified in `self.requirement`
        """
        module = self.requirement.module
        if module in sys.modules:
            return sys.modules[module]
        obj = dynamic_import(module)
        if not isinstance(obj, ModuleType):
            msg = f"Attempted to import {self.requirement.module!r}, expected an instance of ModuleType but got {obj}"
            raise TypeError(msg)
        return obj

    def validate_backend_version(self) -> None:
        """Checks if all of the specified package requirements are met for this Backend.

        Returns: None
        Raises: ValueError: if any of the package requirements are not met.
        """
        messages = [f"{self!r} did not meet the following requirements"]
        validity = []

        for req in self.requirements:
            validity.append(valid := req.is_valid())
            indicator = "\N{HEAVY CHECK MARK}" if valid else "\N{CROSS MARK}"
            messages.append(
                indent(
                    f"{indicator}: {req.module} installed {req.version()} >= {req.min_version}",
                    prefix=" " * 4,
                )
            )

        if not all(validity):
            raise ValueError("\n".join(messages))

    def get_adapter(
        self, cls: type, version: Version = Version.MAIN
    ) -> Adaptation | None:
        """Retrieve the adapter that matches the passed information.

        Arguments:
            cls: type
            version: Version

        Returns:
            Adapter if a match was found. None otherwise.
        """
        for adapt in self.adaptations:
            if adapt.matches(cls=cls, version=version):
                return adapt
        return None


def traverse_rsplits(text: str, sep: str = " ") -> Generator[tuple[str, list[str]]]:
    """Generates all possible rsplits of a string.

    Arguments:
        text: str
        sep: str
            The separator that exists within the text argument

    Yields:
        A partitioning of each of the possible rsplits of the inputted text.

    Examples:
    >>> from narwhals.backends import traverse_rsplits
    >>> list(traverse_rsplits("package.subpackage.module.type", sep="."))
    >>> ("package.subpackage.module", ["type"])
    >>> ("package.subpackage", ["module", "type"])
    >>> ("package", ["subpackage", "module", "type"])
    """
    sep_count = text.count(sep)
    if sep_count == 0:
        yield (text, [])

    for i in range(1, sep_count + 1):
        base, *remaining = text.rsplit(sep, maxsplit=i)
        yield base, remaining


def dynamic_import(dotted_path: str | type, /) -> Any:
    """Attempts to retrieve a specific object specified by a dotted import path.

    Arguments:
        dotted_path: str
            A string that represents a valid Python import.

    Returns:
        The object specified by the import string.

    Examples:
    >>> from narwhals.backends import dynamic_import
    >>> dynamic_import("math.log")
    <built-in function log>
    >>> dynamic_import("os.path.abspath")
    <function abspath at ...>
    """
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


def register_backends(*backends: Backend) -> None:
    """Adds Backend(s) to the global BACKENDS variable."""
    BACKENDS.extendleft(backends)


BACKENDS: deque[Backend] = deque()


register_backends(
    Backend(
        Requirement("pandas", "pandas.__version__", (0, 25, 3)),
        adaptations=[
            Adaptation(
                DataFrame,
                "pandas.DataFrame",
                "narwhals._pandas_like.dataframe.PandasLikeDataFrame",
                level="full",
                version=Version.MAIN | Version.V1,
                kwargs={"validate_column_names": True},
            ),
            Adaptation(
                Series,
                "pandas.Series",
                "narwhals._pandas_like.dataframe.PandasLikeSeries",
                version=Version.MAIN | Version.V1,
                level="full",
            ),
        ],
        implementation=Implementation.PANDAS,
    ),
    Backend(
        Requirement("polars", "polars.__version__", (0, 20, 3)),
        adaptations=[
            Adaptation(
                LazyFrame,
                "polars.LazyFrame",
                "narwhals._polars.dataframe.PolarsLazyFrame",
                version=Version.MAIN | Version.V1,
                level="full",
            ),
            Adaptation(
                DataFrame,
                "polars.DataFrame",
                "narwhals._polars.dataframe.PolarsDataFrame",
                version=Version.MAIN | Version.V1,
                level="full",
            ),
            Adaptation(
                Series,
                "polars.Series",
                "narwhals._polars.series.PolarsSeries",
                version=Version.MAIN | Version.V1,
                level="full",
            ),
        ],
    ),
    Backend(
        Requirement("modin.pandas", "modin.__version__", (0, 25, 3)),
        adaptations=[
            Adaptation(
                DataFrame,
                "modin.pandas.DataFrame",
                "narwhals._pandas_like.dataframe.PandasLikeDataFrame",
                level="full",
                version=Version.MAIN | Version.V1,
                kwargs={"validate_column_names": True},
            ),
            Adaptation(
                Series,
                "modin.pandas.Series",
                "narwhals._pandas_like.dataframe.PandasLikeSeries",
                version=Version.MAIN | Version.V1,
                level="full",
            ),
        ],
        implementation=Implementation.MODIN,
    ),
    Backend(
        Requirement("cudf", "cudf.__version__", (24, 10)),
        adaptations=[
            Adaptation(
                DataFrame,
                "cudf.DataFrame",
                "narwhals._pandas_like.dataframe.PandasLikeDataFrame",
                level="full",
                version=Version.MAIN | Version.V1,
                kwargs={"validate_column_names": True},
            ),
            Adaptation(
                Series,
                "cudf.Series",
                "narwhals._pandas_like.dataframe.PandasLikeSeries",
                version=Version.MAIN | Version.V1,
                level="full",
            ),
        ],
        implementation=Implementation.CUDF,
    ),
    Backend(
        Requirement("pyarrow", "pyarrow.__version__", (11,)),
        adaptations=[
            Adaptation(
                DataFrame,
                "pyarrow.Table",
                "narwhals._arrow.dataframe.ArrowDataFrame",
                level="full",
                version=Version.MAIN | Version.V1,
                kwargs={"validate_column_names": True},
            ),
            Adaptation(
                Series,
                "pyarrow.ChunkedArray",
                "narwhals._arrow.series.ArrowSeries",
                level="full",
                version=Version.MAIN | Version.V1,
                kwargs={"name": ""},
            ),
        ],
    ),
    Backend(
        Requirement("pyspark.sql", "pyspark.__version__", (3, 5)),
        adaptations=[
            Adaptation(
                LazyFrame,
                "pyspark.sql.DataFrame",
                "narwhals._spark_like.dataframe.SparkLikeLazyFrame",
                level="full",
                version=Version.MAIN | Version.V1,
                kwargs={"validate_column_names": True},
            ),
            Adaptation(
                Series,
                "pyspark.sql.Series",
                "narwhals._arrow.dataframe.ArrowSeries",
                level="full",
                version=Version.MAIN | Version.V1,
            ),
        ],
        implementation=Implementation.PYSPARK,
    ),
    Backend(
        Requirement("dask.dataframe", "dask.__version__", (2024, 8)),
        extra_requirements=[
            Requirement("dask_expr", "dask_expr.__version__", (0,)),
        ],
        adaptations=[
            Adaptation(
                LazyFrame,
                "dask.dataframe.DataFrame",
                "narwhals._dask.dataframe.DaskLazyFrame",
                level="full",
                version=Version.MAIN | Version.V1,
                kwargs={"validate_column_names": True},
            ),
            Adaptation(
                LazyFrame,
                "dask_expr.DataFrame",
                "narwhals._dask.dataframe.DaskLazyFrame",
                level="full",
                version=Version.MAIN | Version.V1,
                kwargs={"validate_column_names": True},
            ),
        ],
    ),
    Backend(
        Requirement("duckdb", "duckdb.__version__", (1,)),
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
        Requirement("ibis", "ibis.__version__", (6,)),
        adaptations=[
            Adaptation(
                DataFrame,
                "ibis.expr.types.Table",
                "narwhals._ibis.dataframe.IbisLazyFrame",
                level="full",
                version=Version.V1,
            ),
            Adaptation(
                LazyFrame,
                "ibis.expr.types.Table",
                "narwhals._ibis.dataframe.IbisLazyFrame",
                level="full",
                version=Version.MAIN,
            ),
        ],
    ),
    Backend(
        Requirement("sqlframe", "sqlframe._version.__version__", (3, 22, 0)),
        adaptations=[
            MROAdaptation(
                LazyFrame,
                "sqlframe.base.dataframe.BaseDataFrame",
                "narwhals._spark_like.dataframe.SparkLikeLazyFrame",
                level="full",
                version=Version.MAIN | Version.V1,
                kwargs={"validate_column_names": True},
            ),
        ],
        implementation=Implementation.SQLFRAME,
    ),
)
