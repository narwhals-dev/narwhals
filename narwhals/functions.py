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
from narwhals.translate import from_native
from narwhals.utils import Implementation
from narwhals.utils import validate_laziness

# Missing type parameters for generic type "DataFrame"
# However, trying to provide one results in mypy still complaining...
# The rest of the annotations seem to work fine with this anyway
FrameT = TypeVar("FrameT", bound=Union[DataFrame, LazyFrame])  # type: ignore[type-arg]

if TYPE_CHECKING:
    from types import ModuleType

    from narwhals.dtypes import DType
    from narwhals.schema import Schema
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
    validate_laziness(items)
    first_item = items[0]
    plx = first_item.__narwhals_namespace__()
    return first_item._from_compliant_dataframe(  # type: ignore[return-value]
        plx.concat([df._compliant_frame for df in items], how=how),
    )


def from_dict(
    data: dict[str, Any],
    schema: dict[str, DType] | Schema | None = None,
    *,
    native_namespace: ModuleType | None = None,
) -> DataFrame[Any]:
    """
    Instantiate DataFrame from dictionary.

    Notes:
        For pandas-like dataframes, conversion to schema is applied after dataframe
        creation.

    Arguments:
        data: Dictionary to create DataFrame from.
        schema: The DataFrame schema as Schema or dict of {name: type}.
        native_namespace: The native library to use for DataFrame creation. Only
            necessary if inputs are not Narwhals Series.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals as nw
        >>> data = {"a": [1, 2, 3], "b": [4, 5, 6]}

        Let's define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df):
        ...     data = {"c": [5, 2], "d": [1, 4]}
        ...     native_namespace = nw.get_native_namespace(df)
        ...     return nw.from_dict(data, native_namespace=native_namespace)

        Let's see what happens when passing pandas / Polars input:

        >>> func(pd.DataFrame(data))
           c  d
        0  5  1
        1  2  4
        >>> func(pl.DataFrame(data))
        shape: (2, 2)
        ┌─────┬─────┐
        │ c   ┆ d   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 5   ┆ 1   │
        │ 2   ┆ 4   │
        └─────┴─────┘
    """
    from narwhals.series import Series
    from narwhals.translate import to_native

    if not data:
        msg = "from_dict cannot be called with empty dictionary"
        raise ValueError(msg)
    if native_namespace is None:
        for val in data.values():
            if isinstance(val, Series):
                native_namespace = val.__native_namespace__()
                break
        else:
            msg = "Calling `from_dict` without `native_namespace` is only supported if all input values are already Narwhals Series"
            raise TypeError(msg)
        data = {key: to_native(value, strict=False) for key, value in data.items()}
    implementation = Implementation.from_native_namespace(native_namespace)

    if implementation is Implementation.POLARS:
        if schema:
            from narwhals._polars.utils import (
                reverse_translate_dtype as polars_reverse_translate_dtype,
            )

            schema = {
                name: polars_reverse_translate_dtype(dtype)
                for name, dtype in schema.items()
            }

        native_frame = native_namespace.from_dict(data, schema=schema)
    elif implementation in {
        Implementation.PANDAS,
        Implementation.MODIN,
        Implementation.CUDF,
    }:
        native_frame = native_namespace.DataFrame.from_dict(data)

        if schema:
            from narwhals._pandas_like.utils import (
                reverse_translate_dtype as pandas_like_reverse_translate_dtype,
            )

            schema = {
                name: pandas_like_reverse_translate_dtype(
                    schema[name], native_type, implementation
                )
                for name, native_type in native_frame.dtypes.items()
            }
            native_frame = native_frame.astype(schema)

    elif implementation is Implementation.PYARROW:
        if schema:
            from narwhals._arrow.utils import (
                reverse_translate_dtype as arrow_reverse_translate_dtype,
            )

            schema = native_namespace.schema(
                [
                    (name, arrow_reverse_translate_dtype(dtype))
                    for name, dtype in schema.items()
                ]
            )
        native_frame = native_namespace.table(data, schema=schema)
    else:  # pragma: no cover
        try:
            # implementation is UNKNOWN, Narhwals extension using this feature should
            # implement `from_dict` function in the top-level namespace.
            native_frame = native_namespace.from_dict(data)
        except AttributeError as e:
            msg = "Unknown namespace is expected to implement `from_dict` function."
            raise AttributeError(msg) from e
    return from_native(native_frame, eager_only=True)


def _get_sys_info() -> dict[str, str]:
    """System information

    Returns system and Python version information

    Copied from sklearn

    """
    python = sys.version.replace("\n", " ")

    blob = (
        ("python", python),
        ("executable", sys.executable),
        ("machine", platform.platform()),
    )

    return dict(blob)


def _get_deps_info() -> dict[str, str]:
    """Overview of the installed version of main dependencies

    This function does not import the modules to collect the version numbers
    but instead relies on standard Python package metadata.

    Returns version information on relevant Python libraries

    This function and show_versions were copied from sklearn and adapted

    """
    deps = (
        "pandas",
        "polars",
        "cudf",
        "modin",
        "pyarrow",
        "numpy",
    )

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
