import os
import sys

import narwhals as nw
from narwhals.utils import remove_prefix
from narwhals.utils import remove_suffix

# todo: make dtypes reference page as well
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
    print("not documented")  # noqa: T201
    print(missing)  # noqa: T201
    sys.exit(1)
if extra := set(documented).difference(top_level_functions):
    print("outdated")  # noqa: T201
    print(extra)  # noqa: T201
    sys.exit(1)
sys.exit(0)
