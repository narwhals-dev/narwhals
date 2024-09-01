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
    """
    Concatenate multiple DataFrames, LazyFrames into a single entity.

    Arguments:
        items: DataFrames, LazyFrames to concatenate.

        how: {'vertical', 'horizontal'}
            * vertical: Stacks Series from DataFrames vertically and fills with `null`
              if the lengths don't match.
            * horizontal: Stacks Series from DataFrames horizontally and fills with `null`
              if the lengths don't match.

    Returns:
        A new DataFrame, Lazyframe resulting from the concatenation.

    Raises:
        NotImplementedError: The items to concatenate should either all be eager, or all lazy

    Examples:

        Let's take an example of vertical concatenation:

        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals as nw
        >>> data_1 = {"a": [1, 2, 3], "b": [4, 5, 6]}
        >>> data_2 = {"a": [5, 2], "b": [1, 4]}

        >>> df_pd_1 = pd.DataFrame(data_1)
        >>> df_pd_2 = pd.DataFrame(data_2)
        >>> df_pl_1 = pl.DataFrame(data_1)
        >>> df_pl_2 = pl.DataFrame(data_2)

        Let's define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df1, df2):
        ...     return nw.concat([df1, df2], how="vertical")

        >>> func(df_pd_1, df_pd_2)
           a  b
        0  1  4
        1  2  5
        2  3  6
        0  5  1
        1  2  4
        >>> func(df_pl_1, df_pl_2)
        shape: (5, 2)
        ┌─────┬─────┐
        │ a   ┆ b   │
        │ --- ┆ --- │
        │ i64 ┆ i64 │
        ╞═════╪═════╡
        │ 1   ┆ 4   │
        │ 2   ┆ 5   │
        │ 3   ┆ 6   │
        │ 5   ┆ 1   │
        │ 2   ┆ 4   │
        └─────┴─────┘

        Let's look at case a for horizontal concatenation:

        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals as nw
        >>> data_1 = {"a": [1, 2, 3], "b": [4, 5, 6]}
        >>> data_2 = {"c": [5, 2], "d": [1, 4]}

        >>> df_pd_1 = pd.DataFrame(data_1)
        >>> df_pd_2 = pd.DataFrame(data_2)
        >>> df_pl_1 = pl.DataFrame(data_1)
        >>> df_pl_2 = pl.DataFrame(data_2)

        Defining a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df1, df2):
        ...     return nw.concat([df1, df2], how="horizontal")

        >>> func(df_pd_1, df_pd_2)
           a  b    c    d
        0  1  4  5.0  1.0
        1  2  5  2.0  4.0
        2  3  6  NaN  NaN

        >>> func(df_pl_1, df_pl_2)
        shape: (3, 4)
        ┌─────┬─────┬──────┬──────┐
        │ a   ┆ b   ┆ c    ┆ d    │
        │ --- ┆ --- ┆ ---  ┆ ---  │
        │ i64 ┆ i64 ┆ i64  ┆ i64  │
        ╞═════╪═════╪══════╪══════╡
        │ 1   ┆ 4   ┆ 5    ┆ 1    │
        │ 2   ┆ 5   ┆ 2    ┆ 4    │
        │ 3   ┆ 6   ┆ null ┆ null │
        └─────┴─────┴──────┴──────┘

    """

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


def new_series(
    name: str,
    values: Any,
    dtype: DType | type[DType] | None = None,
    *,
    native_namespace: ModuleType,
) -> Series:
    """
    Instantiate Narwhals Series from raw data.

    Arguments:
        name: Name of resulting Series.
        values: Values of make Series from.
        dtype: (Narwhals) dtype. If not provided, the native library
            may auto-infer it from `values`.
        native_namespace: The native library to use for DataFrame creation.

    Examples:
        >>> import pandas as pd
        >>> import polars as pl
        >>> import narwhals as nw
        >>> data = {"a": [1, 2, 3], "b": [4, 5, 6]}

        Let's define a dataframe-agnostic function:

        >>> @nw.narwhalify
        ... def func(df):
        ...     values = [4, 1, 2]
        ...     native_namespace = nw.get_native_namespace(df)
        ...     return nw.new_series("c", values, nw.Int32, native_namespace=native_namespace)

        Let's see what happens when passing pandas / Polars input:

        >>> func(pd.DataFrame(data))
        0    4
        1    1
        2    2
        Name: c, dtype: int32
        >>> func(pl.DataFrame(data))  # doctest: +NORMALIZE_WHITESPACE
        shape: (3,)
        Series: 'c' [i32]
        [
           4
           1
           2
        ]
    """
    implementation = Implementation.from_native_namespace(native_namespace)

    if implementation is Implementation.POLARS:
        if dtype:
            from narwhals._polars.utils import (
                narwhals_to_native_dtype as polars_narwhals_to_native_dtype,
            )

            dtype = polars_narwhals_to_native_dtype(dtype)

        native_series = native_namespace.Series(name=name, values=values, dtype=dtype)
    elif implementation in {
        Implementation.PANDAS,
        Implementation.MODIN,
        Implementation.CUDF,
    }:
        if dtype:
            from narwhals._pandas_like.utils import (
                narwhals_to_native_dtype as pandas_like_narwhals_to_native_dtype,
            )

            dtype = pandas_like_narwhals_to_native_dtype(dtype, None, implementation)
        native_series = native_namespace.Series(values, name=name, dtype=dtype)

    elif implementation is Implementation.PYARROW:
        if dtype:
            from narwhals._arrow.utils import (
                narwhals_to_native_dtype as arrow_narwhals_to_native_dtype,
            )

            dtype = arrow_narwhals_to_native_dtype(dtype)
        native_series = native_namespace.chunked_array([values], type=dtype)

    elif implementation is Implementation.DASK:
        msg = "Dask support in Narwhals is lazy-only, so `new_series` is " "not supported"
        raise NotImplementedError(msg)
    else:  # pragma: no cover
        try:
            # implementation is UNKNOWN, Narwhals extension using this feature should
            # implement `from_dict` function in the top-level namespace.
            native_series = native_namespace.new_series(name, values, dtype)
        except AttributeError as e:
            msg = "Unknown namespace is expected to implement `Series` constructor."
            raise AttributeError(msg) from e
    return from_native(native_series, series_only=True).alias(name)


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

        Let's create a new dataframe of the same class as the dataframe we started with, from a dict of new data:

        >>> @nw.narwhalify
        ... def func(df):
        ...     new_data = {"c": [5, 2], "d": [1, 4]}
        ...     native_namespace = nw.get_native_namespace(df)
        ...     return nw.from_dict(new_data, native_namespace=native_namespace)

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
                narwhals_to_native_dtype as polars_narwhals_to_native_dtype,
            )

            schema = {
                name: polars_narwhals_to_native_dtype(dtype)
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
                narwhals_to_native_dtype as pandas_like_narwhals_to_native_dtype,
            )

            schema = {
                name: pandas_like_narwhals_to_native_dtype(
                    schema[name], native_type, implementation
                )
                for name, native_type in native_frame.dtypes.items()
            }
            native_frame = native_frame.astype(schema)

    elif implementation is Implementation.PYARROW:
        if schema:
            from narwhals._arrow.utils import (
                narwhals_to_native_dtype as arrow_narwhals_to_native_dtype,
            )

            schema = native_namespace.schema(
                [
                    (name, arrow_narwhals_to_native_dtype(dtype))
                    for name, dtype in schema.items()
                ]
            )
        native_frame = native_namespace.table(data, schema=schema)
    else:  # pragma: no cover
        try:
            # implementation is UNKNOWN, Narwhals extension using this feature should
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
