from __future__ import annotations

import inspect
import string
import sys
from inspect import isfunction
from pathlib import Path
from typing import Any
from typing import Iterator

import polars as pl

import narwhals as nw
from narwhals.utils import remove_prefix

LOWERCASE = tuple(string.ascii_lowercase)

if sys.version_info >= (3, 13):

    def _is_public_method_or_property(obj: Any) -> bool:
        return (isfunction(obj) or isinstance(obj, property)) and obj.__name__.startswith(
            LOWERCASE
        )
else:

    def _is_public_method_or_property(obj: Any) -> bool:
        return (isfunction(obj) and obj.__name__.startswith(LOWERCASE)) or (
            isinstance(obj, property) and obj.fget.__name__.startswith(LOWERCASE)
        )


def iter_api_reference_names(tp: type[Any]) -> Iterator[str]:
    for name, _ in inspect.getmembers(tp, _is_public_method_or_property):
        yield name


ret = 0

NAMESPACES = {"dt", "str", "cat", "name", "list", "struct"}
EXPR_ONLY_METHODS = {"over", "map_batches"}
SERIES_ONLY_METHODS = {
    "dtype",
    "implementation",
    "is_empty",
    "is_sorted",
    "hist",
    "item",
    "name",
    "rename",
    "scatter",
    "shape",
    "to_arrow",
    "to_dummies",
    "to_frame",
    "to_list",
    "to_native",
    "to_numpy",
    "to_pandas",
    "to_polars",
    "value_counts",
    "zip_with",
    "__iter__",
    "__contains__",
}
BASE_DTYPES = {
    "NumericType",
    "DType",
    "TemporalType",
    "Literal",
    "OrderedDict",
    "Mapping",
    "Iterable",
}

files = {fp.stem for fp in Path("narwhals").iterdir()}

# Top level functions
top_level_functions = [
    i for i in dir(nw) if not i[0].isupper() and i[0] != "_" and i not in files
]
with open("docs/api-reference/narwhals.md") as fd:
    content = fd.read()
documented = [
    remove_prefix(i, "        - ")
    for i in content.splitlines()
    if i.startswith("        - ")
]
if missing := set(top_level_functions).difference(documented).difference({"annotations"}):
    print("top-level functions: not documented")  # noqa: T201
    print(missing)  # noqa: T201
    ret = 1
if extra := set(documented).difference(top_level_functions):
    print("top-level functions: outdated")  # noqa: T201
    print(extra)  # noqa: T201
    ret = 1

# DataFrame methods
dataframe_methods = list(iter_api_reference_names(nw.DataFrame))
with open("docs/api-reference/dataframe.md") as fd:
    content = fd.read()
documented = [
    remove_prefix(i, "        - ")
    for i in content.splitlines()
    if i.startswith("        - ") and not i.startswith("        - _")
]
if missing := set(dataframe_methods).difference(documented):
    print("DataFrame: not documented")  # noqa: T201
    print(missing)  # noqa: T201
    ret = 1
if extra := set(documented).difference(dataframe_methods):
    print("DataFrame: outdated")  # noqa: T201
    print(extra)  # noqa: T201
    ret = 1

# LazyFrame methods
lazyframe_methods = list(iter_api_reference_names(nw.LazyFrame))
with open("docs/api-reference/lazyframe.md") as fd:
    content = fd.read()
documented = [
    remove_prefix(i, "        - ")
    for i in content.splitlines()
    if i.startswith("        - ")
]
if missing := set(lazyframe_methods).difference(documented):
    print("LazyFrame: not documented")  # noqa: T201
    print(missing)  # noqa: T201
    ret = 1
if extra := set(documented).difference(lazyframe_methods):
    print("LazyFrame: outdated")  # noqa: T201
    print(extra)  # noqa: T201
    ret = 1

# Series methods
series_methods = list(iter_api_reference_names(nw.Series))
with open("docs/api-reference/series.md") as fd:
    content = fd.read()
documented = [
    remove_prefix(i, "        - ")
    for i in content.splitlines()
    if i.startswith("        - ") and not i.startswith("        - _")
]
if missing := set(series_methods).difference(documented).difference(NAMESPACES):
    print("Series: not documented")  # noqa: T201
    print(missing)  # noqa: T201
    ret = 1
if extra := set(documented).difference(series_methods):
    print("Series: outdated")  # noqa: T201
    print(extra)  # noqa: T201
    ret = 1

# Series.{cat, dt, list, str} methods
for namespace in NAMESPACES.difference({"name"}):
    series_ns_methods = [
        i
        for i in dir(getattr(nw.from_native(pl.Series(), series_only=True), namespace))
        if not i[0].isupper() and i[0] != "_"
    ]
    with open(f"docs/api-reference/series_{namespace}.md") as fd:
        content = fd.read()
    documented = [
        remove_prefix(i, "        - ")
        for i in content.splitlines()
        if i.startswith("        - ") and not i.startswith("        - _")
    ]
    if missing := set(series_ns_methods).difference(documented):
        print(f"Series.{namespace}: not documented")  # noqa: T201
        print(missing)  # noqa: T201
        ret = 1
    if extra := set(documented).difference(series_ns_methods):
        print(f"Series.{namespace}: outdated")  # noqa: T201
        print(extra)  # noqa: T201
        ret = 1

# Expr methods
expr_methods = list(iter_api_reference_names(nw.Expr))
with open("docs/api-reference/expr.md") as fd:
    content = fd.read()
documented = [
    remove_prefix(i, "        - ")
    for i in content.splitlines()
    if i.startswith("        - ")
]
if missing := set(expr_methods).difference(documented).difference(NAMESPACES):
    print("Expr: not documented")  # noqa: T201
    print(missing)  # noqa: T201
    ret = 1
if extra := set(documented).difference(expr_methods):
    print("Expr: outdated")  # noqa: T201
    print(extra)  # noqa: T201
    ret = 1

# Expr.{cat, dt, list, name, str} methods
for namespace in NAMESPACES:
    expr_ns_methods = [
        i
        for i in dir(getattr(nw.col("a"), namespace))
        if not i[0].isupper() and i[0] != "_"
    ]
    with open(f"docs/api-reference/expr_{namespace}.md") as fd:
        content = fd.read()
    documented = [
        remove_prefix(i, "        - ")
        for i in content.splitlines()
        if i.startswith("        - ")
    ]
    if missing := set(expr_ns_methods).difference(documented):
        print(f"Expr.{namespace}: not documented")  # noqa: T201
        print(missing)  # noqa: T201
        ret = 1
    if extra := set(documented).difference(expr_ns_methods):
        print(f"Expr.{namespace}: outdated")  # noqa: T201
        print(extra)  # noqa: T201
        ret = 1

# DTypes
dtypes = [i for i in dir(nw.dtypes) if i[0].isupper() and not i.isupper() and i[0] != "_"]
with open("docs/api-reference/dtypes.md") as fd:
    content = fd.read()
documented = [
    remove_prefix(i, "        - ")
    for i in content.splitlines()
    if i.startswith("        - ") and not i.startswith("        - _")
]
if missing := set(dtypes).difference(documented).difference(BASE_DTYPES):
    print("Dtype: not documented")  # noqa: T201
    print(missing)  # noqa: T201
    ret = 1
if extra := set(documented).difference(dtypes):
    print("Dtype: outdated")  # noqa: T201
    print(extra)  # noqa: T201
    ret = 1

# Check Expr vs Series
if missing := set(expr_methods).difference(series_methods).difference(EXPR_ONLY_METHODS):
    print("In Expr but not in Series")  # noqa: T201
    print(missing)  # noqa: T201
    ret = 1
if extra := set(series_methods).difference(expr_methods).difference(SERIES_ONLY_METHODS):
    print("In Series but not in Expr")  # noqa: T201
    print(extra)  # noqa: T201
    ret = 1

# Check Expr vs Series internal methods
for namespace in NAMESPACES.difference({"name"}):
    expr_internal = [
        i
        for i in dir(getattr(nw.col("a"), namespace))
        if not i[0].isupper() and i[0] != "_"
    ]
    series_internal = [
        i
        for i in dir(getattr(nw.from_native(pl.Series(), series_only=True), namespace))
        if not i[0].isupper() and i[0] != "_"
    ]
    if missing := set(expr_internal).difference(series_internal):
        print(f"In Expr.{namespace} but not in Series.{namespace}")  # noqa: T201
        print(missing)  # noqa: T201
        ret = 1
    if extra := set(series_internal).difference(expr_internal):
        print(f"In Series.{namespace} but not in Expr.{namespace}")  # noqa: T201
        print(extra)  # noqa: T201
        ret = 1

sys.exit(ret)
