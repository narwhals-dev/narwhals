# ruff: noqa
# type: ignore
import sys

sys.path.append("..")

import pandas as pd
import polars as pl

pd_series = pd.Series([1], name="a").__column_consortium_standard__()
pl_series = pl.Series("a", [1]).__column_consortium_standard__()
pd_df = pd.DataFrame({"a": [1]}).__dataframe_consortium_standard__()
pl_df = pl.DataFrame({"a": [1]}).__dataframe_consortium_standard__()
pd_scalar = pd_df.col("a").mean()
pl_scalar = pl_df.col("a").mean()
pd_namespace = pd_df.__dataframe_namespace__()
pl_namespace = pl_df.__dataframe_namespace__()

for name, object in [
    ("pandas-column.md", pd_series),
    ("polars-column.md", pl_series),
    ("pandas-dataframe.md", pd_df),
    ("polars-dataframe.md", pl_df),
    ("pandas-scalar.md", pd_scalar),
    ("polars-scalar.md", pl_scalar),
    ("pandas-namespace.md", pd_scalar),
    ("polars-namespace.md", pl_scalar),
]:
    members = [
        i for i in object.__dir__() if not (i.startswith("_") and not i.startswith("__"))
    ]

    with open(name) as fd:
        content = fd.read()

    members_txt = "\n      - ".join(sorted(members)) + "\n      "

    start = content.index("members:")
    end = content.index("show_signature")
    content = content[:start] + f"members:\n      - {members_txt}" + content[end:]

    with open(name, "w") as fd:
        fd.write(content)
