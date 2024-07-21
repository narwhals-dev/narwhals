from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any
from typing import Final

import polars as pl
from jinja2 import Template

TEMPLATE_PATH: Final[Path] = Path("utils") / "api-completeness.md.jinja"
DESTINATION_PATH: Final[Path] = Path("docs") / "api-completeness.md"


MODULES = ["dataframe", "series", "expr"]
EXCLUDE_CLASSES = {"BaseFrame"}


def get_class_methods(kls: type[Any]) -> list[str]:
    return [m[0] for m in inspect.getmembers(kls) if not m[0].startswith("_")]


def get_backend_completeness_table() -> str:
    results = []

    for module_name in MODULES:
        nw_namespace = f"narwhals.{module_name}"
        sub_module_name = module_name

        narwhals_module_ = importlib.import_module(nw_namespace)
        classes_ = inspect.getmembers(
            narwhals_module_,
            predicate=lambda c: inspect.isclass(c) and c.__module__ == nw_namespace,  # noqa: B023, not imported classes
        )

        for nw_class_name, nw_class in classes_:
            if nw_class_name in EXCLUDE_CLASSES:
                continue
            if nw_class_name == "LazyFrame":
                backend_class_name = "DataFrame"
            else:
                backend_class_name = nw_class_name

            arrow_class_name = f"Arrow{backend_class_name}"
            arrow_module_ = importlib.import_module(f"narwhals._arrow.{sub_module_name}")
            arrow_class = inspect.getmembers(
                arrow_module_,
                predicate=lambda c: inspect.isclass(c) and c.__name__ == arrow_class_name,  # noqa: B023
            )

            pandas_class_name = f"PandasLike{backend_class_name}"
            pandas_module_ = importlib.import_module(
                f"narwhals._pandas_like.{sub_module_name}"
            )
            pandas_class = inspect.getmembers(
                pandas_module_,
                predicate=lambda c: inspect.isclass(c)
                and c.__name__ == pandas_class_name,  # noqa: B023
            )

            nw_methods = get_class_methods(nw_class)
            arrow_methods = get_class_methods(arrow_class[0][1]) if arrow_class else []
            pandas_methods = get_class_methods(pandas_class[0][1]) if pandas_class else []

            narhwals = pl.DataFrame(
                {"Class": nw_class_name, "Backend": "narwhals", "Method": nw_methods}
            )
            arrow = pl.DataFrame(
                {"Class": nw_class_name, "Backend": "arrow", "Method": arrow_methods}
            )
            pandas = pl.DataFrame(
                {
                    "Class": nw_class_name,
                    "Backend": "pandas-like",
                    "Method": pandas_methods,
                }
            )

            results.extend([narhwals, pandas, arrow])

    results = (
        pl.concat(results)  # noqa: PD010
        .with_columns(supported=pl.lit(":white_check_mark:"))
        .pivot(on="Backend", values="supported", index=["Class", "Method"])
        .filter(pl.col("narwhals").is_not_null())
        .drop("narwhals")
        .fill_null(":x:")
        .sort("Class", "Method")
    )

    with pl.Config(
        tbl_formatting="ASCII_MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
        set_tbl_rows=results.shape[0],
    ):
        return str(results)


backend_table = get_backend_completeness_table()

with TEMPLATE_PATH.open(mode="r") as stream:
    new_content = Template(stream.read()).render({"backend_table": backend_table})

with DESTINATION_PATH.open(mode="w") as destination:
    destination.write(new_content)
