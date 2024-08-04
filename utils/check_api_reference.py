import os
import sys

import polars as pl

import narwhals as nw
from narwhals.utils import remove_prefix
from narwhals.utils import remove_suffix

ret = 0

# TODO(Unassigned): make dtypes reference page as well
files = {remove_suffix(i, ".py") for i in os.listdir("narwhals")}
top_level_functions = [
    i
    for i in nw.__dir__()
    if not i[0].isupper()
    and i[0] != "_"
    and i not in files
    and i not in {"annotations", "DataFrame", "LazyFrame", "Series"}
]
with open("docs/api-reference/narwhals.md") as fd:
    content = fd.read()
documented = [
    remove_prefix(i, "        - ")
    for i in content.splitlines()
    if i.startswith("        - ")
]
if missing := set(top_level_functions).difference(documented):
    print("top-level functions: not documented")  # noqa: T201
    print(missing)  # noqa: T201
    ret = 1
if extra := set(documented).difference(top_level_functions):
    print("top-level functions: outdated")  # noqa: T201
    print(extra)  # noqa: T201
    ret = 1

top_level_functions = [
    i
    for i in nw.from_native(pl.DataFrame()).__dir__()
    if not i[0].isupper() and i[0] != "_"
]
with open("docs/api-reference/dataframe.md") as fd:
    content = fd.read()
documented = [
    remove_prefix(i, "        - ")
    for i in content.splitlines()
    if i.startswith("        - ")
]
if missing := set(top_level_functions).difference(documented):
    print("DataFrame: not documented")  # noqa: T201
    print(missing)  # noqa: T201
    ret = 1
if extra := set(documented).difference(top_level_functions).difference({"__getitem__"}):
    print("DataFrame: outdated")  # noqa: T201
    print(extra)  # noqa: T201
    ret = 1

top_level_functions = [
    i
    for i in nw.from_native(pl.LazyFrame()).__dir__()
    if not i[0].isupper() and i[0] != "_"
]
with open("docs/api-reference/lazyframe.md") as fd:
    content = fd.read()
documented = [
    remove_prefix(i, "        - ")
    for i in content.splitlines()
    if i.startswith("        - ")
]
if missing := set(top_level_functions).difference(documented):
    print("LazyFrame: not documented")  # noqa: T201
    print(missing)  # noqa: T201
    ret = 1
if extra := set(documented).difference(top_level_functions):
    print("LazyFrame: outdated")  # noqa: T201
    print(extra)  # noqa: T201
    ret = 1

top_level_functions = [
    i
    for i in nw.from_native(pl.Series(), series_only=True).__dir__()
    if not i[0].isupper() and i[0] != "_"
]
with open("docs/api-reference/series.md") as fd:
    content = fd.read()
documented = [
    remove_prefix(i, "        - ")
    for i in content.splitlines()
    if i.startswith("        - ")
]
if (
    missing := set(top_level_functions)
    .difference(documented)
    .difference({"dt", "str", "cat", "name"})
):
    print("Series: not documented")  # noqa: T201
    print(missing)  # noqa: T201
    ret = 1
if extra := set(documented).difference(top_level_functions):
    print("Series: outdated")  # noqa: T201
    print(extra)  # noqa: T201
    ret = 1

top_level_functions = [
    i for i in nw.Expr(lambda: 0).__dir__() if not i[0].isupper() and i[0] != "_"
]
with open("docs/api-reference/expr.md") as fd:
    content = fd.read()
documented = [
    remove_prefix(i, "        - ")
    for i in content.splitlines()
    if i.startswith("        - ")
]
if (
    missing := set(top_level_functions)
    .difference(documented)
    .difference({"cat", "str", "dt", "name"})
):
    print("Expr: not documented")  # noqa: T201
    print(missing)  # noqa: T201
    ret = 1
if extra := set(documented).difference(top_level_functions):
    print("Expr: outdated")  # noqa: T201
    print(extra)  # noqa: T201
    ret = 1

# DTypes

# dt

# str

# Check Expr vs Series
expr = [i for i in nw.Expr(lambda: 0).__dir__() if not i[0].isupper() and i[0] != "_"]
series = [
    i
    for i in nw.from_native(pl.Series(), series_only=True).__dir__()
    if not i[0].isupper() and i[0] != "_"
]

if missing := set(expr).difference(series).difference({"over"}):
    print("In expr but not in series")  # noqa: T201
    print(missing)  # noqa: T201
    ret = 1
if (
    extra := set(series)
    .difference(expr)
    .difference(
        {
            "to_dummies",
            "to_pandas",
            "to_list",
            "to_numpy",
            "dtype",
            "name",
            "shape",
            "to_frame",
            "is_empty",
            "is_sorted",
            "value_counts",
            "zip_with",
            "item",
        }
    )
):
    print("in series but not in expr")  # noqa: T201
    print(extra)  # noqa: T201
    ret = 1

sys.exit(ret)
