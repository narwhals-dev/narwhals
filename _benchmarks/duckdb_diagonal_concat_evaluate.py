# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars==1.37.0",
# ]
# ///
from __future__ import annotations

from pathlib import Path

import polars as pl

branch = pl.read_csv("runtime-branch.csv").rename({"time": "time_branch"})
main = pl.read_csv("runtime-main.csv").rename({"time": "time_main"})
frame = (
    branch.join(main, on=["n_frames", "n_rows", "n_cols"], how="left")
    .with_columns(
        pl.col("time_main"),
        pl.col("time_branch"),
        diff=(pl.col("time_main") - pl.col("time_branch")),
        speedup=(pl.col("time_main") / pl.col("time_branch")),
    )
    .with_columns(pl.selectors.float().round(3))
    .rename(
        {
            "n_frames": "n frames",
            "n_rows": "n rows",
            "n_cols": "n cols",
            "time_main": "time[s] (main)",
            "time_branch": "time[s] (branch)",
            "diff": "diff[s] (main - branch)",
            "speedup": "speedup (main / branch)",
        }
    )
)

path = Path("duckdb-diagonal-concat-evaluation.md")
config = pl.Config(
    tbl_rows=100,
    tbl_cols=100,
    tbl_formatting="MARKDOWN",
    tbl_hide_column_data_types=True,
    tbl_hide_dataframe_shape=True,
    tbl_cell_alignment="LEFT",
    tbl_width_chars=-1,
)

with config, path.open(mode="w", encoding="utf-8") as file:
    file.write(str(frame))
