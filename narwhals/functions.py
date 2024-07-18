from __future__ import annotations

import platform
import sys
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Literal
from typing import TypeVar
from typing import Union

from narwhals.dataframe import DataFrame
from narwhals.dataframe import LazyFrame
from narwhals.utils import validate_laziness
from narwhals.utils import validate_same_library

# Missing type parameters for generic type "DataFrame"
# However, trying to provide one results in mypy still complaining...
# The rest of the annotations seem to work fine with this anyway
FrameT = TypeVar("FrameT", bound=Union[DataFrame, LazyFrame])  # type: ignore[type-arg]

if TYPE_CHECKING:
    from narwhals.series import Series


def concat(
    items: Iterable[FrameT],
    *,
    how: Literal["horizontal", "vertical"] = "vertical",
) -> FrameT:
    if how not in ("horizontal", "vertical"):  # pragma: no cover
        msg = "Only horizontal and vertical concatenations are supported"
        raise NotImplementedError(msg)
    if not items:
        msg = "No items to concatenate"
        raise ValueError(msg)
    items = list(items)
    validate_same_library(items)
    validate_laziness(items)
    first_item = items[0]
    plx = first_item.__narwhals_namespace__()
    return first_item._from_compliant_dataframe(  # type: ignore[return-value]
        plx.concat([df._compliant_frame for df in items], how=how),
    )


def _get_sys_info() -> dict[str, str]:
    """System information

    Returns system and Python version information

    Copied from sklearn

    """
    python = sys.version.replace("\n", " ")

    blob = [
        ("python", python),
        ("executable", sys.executable),
        ("machine", platform.platform()),
    ]

    return dict(blob)


def _get_deps_info() -> dict[str, str]:
    """Overview of the installed version of main dependencies

    This function does not import the modules to collect the version numbers
    but instead relies on standard Python package metadata.

    Returns version information on relevant Python libraries

    This function and show_versions were copied from sklearn and adapted

    """
    deps = [
        "pandas",
        "polars",
        "cudf",
        "modin",
        "pyarrow",
        "numpy",
    ]

    from . import __version__

    deps_info = {
        "narwhals": __version__,
    }

    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version

    for modname in deps:
        try:
            deps_info[modname] = version(modname)
        except PackageNotFoundError:  # noqa: PERF203
            deps_info[modname] = ""
    return deps_info


def show_versions() -> None:
    """
    Print useful debugging information

    Examples:

        >>> from narwhals import show_versions
        >>> show_versions()  # doctest:+SKIP
    """

    sys_info = _get_sys_info()
    deps_info = _get_deps_info()

    print("\nSystem:")  # noqa: T201
    for k, stat in sys_info.items():
        print(f"{k:>10}: {stat}")  # noqa: T201

    print("\nPython dependencies:")  # noqa: T201
    for k, stat in deps_info.items():
        print(f"{k:>13}: {stat}")  # noqa: T201


def get_level(
    obj: DataFrame[Any] | LazyFrame[Any] | Series,
) -> Literal["full", "interchange"]:
    """
    Level of support Narwhals has for current object.

    This can be one of:

    - 'full': full Narwhals API support
    - 'metadata': only metadata operations are supported (`df.schema`)
    """
    return obj._level
