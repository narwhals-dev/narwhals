from __future__ import annotations

import importlib
import inspect
from enum import Enum
from enum import auto
from pathlib import Path
from typing import Any
from typing import Final
from typing import NamedTuple

import polars as pl
from jinja2 import Template

TEMPLATE_PATH: Final[Path] = Path("utils") / "api-completeness.md.jinja"
DESTINATION_PATH: Final[Path] = Path("docs") / "api-completeness"


class BackendType(Enum):
    LAZY = auto()
    EAGER = auto()
    BOTH = auto()


class Backend(NamedTuple):
    name: str
    module: str
    type_: BackendType


MODULES = ["dataframe", "series", "expr"]

BACKENDS = [
    Backend(name="arrow", module="_arrow", type_=BackendType.EAGER),
    Backend(name="dask", module="_dask", type_=BackendType.LAZY),
    Backend(name="duckdb", module="_dask", type_=BackendType.LAZY),
    Backend(name="pandas-like", module="_pandas_like", type_=BackendType.EAGER),
    Backend(name="spark-like", module="_spark_like", type_=BackendType.LAZY),
]

EXCLUDE_CLASSES = {"BaseFrame", "Then", "When"}

DIRECTLY_IMPLEMENTED_METHODS = ["pipe", "implementation", "to_native"]


def get_class_methods(kls: type[Any]) -> list[str]:
    return [m[0] for m in inspect.getmembers(kls) if not m[0].startswith("_")]


def parse_module(module_name: str, backend: str, nw_class_name: str) -> list[str]:
    try:
        module_ = importlib.import_module(f"narwhals.{backend}.{module_name}")
        class_ = inspect.getmembers(
            module_,
            predicate=lambda c: (
                inspect.isclass(c)
                and c.__name__.endswith(nw_class_name)
                and not c.__name__.startswith("Compliant")  # Exclude protocols
                and not c.__name__.startswith("DuckDBInterchange")
            ),
        )

        methods_ = (
            get_class_methods(class_[0][1]) + DIRECTLY_IMPLEMENTED_METHODS
            if class_
            else []
        )

    except ModuleNotFoundError:
        methods_ = []

    return methods_


def render_table_and_write_to_output(
    results: list[pl.DataFrame], title: str, output_filename: str
) -> None:
    results = (
        pl.concat(results)
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
        set_tbl_width_chars=1_000,
    ):
        table = str(results)

    with TEMPLATE_PATH.open(mode="r") as stream:
        new_content = Template(stream.read()).render(
            {"backend_table": table, "title": title}
        )

    with (DESTINATION_PATH / f"{output_filename}.md").open(mode="w") as destination:
        destination.write(new_content)

    return table


def get_backend_completeness_table() -> None:
    for module_name in MODULES:
        results = []
        processed_classes = set()

        nw_namespace = f"narwhals.{module_name}"

        narwhals_module_ = importlib.import_module(nw_namespace)
        classes_ = inspect.getmembers(
            narwhals_module_,
            predicate=lambda c: inspect.isclass(c) and c.__module__ == nw_namespace,  # noqa: B023, not imported classes
        )

        for nw_class_name, nw_class in classes_:
            if nw_class_name in EXCLUDE_CLASSES:
                continue

            nw_methods = get_class_methods(nw_class)

            narwhals = pl.DataFrame(
                {"Class": nw_class_name, "Backend": "narwhals", "Method": nw_methods}
            )

            backend_methods = [
                pl.DataFrame(
                    {
                        "Class": nw_class_name,
                        "Backend": backend.name,
                        "Method": parse_module(
                            module_name,
                            backend=backend.module,
                            nw_class_name=nw_class_name,
                        ),
                        # "Type": backend.type_
                    }
                )
                for backend in BACKENDS
            ]

            results.extend([narwhals, *backend_methods])

            if nw_class_name in {"DataFrame", "LazyFrame"}:
                render_table_and_write_to_output(
                    results=[narwhals, *backend_methods],
                    title=nw_class_name,
                    output_filename=nw_class_name.lower(),
                )
                processed_classes.add(nw_class_name)

        if processed_classes == {"DataFrame", "LazyFrame"}:
            continue

        render_table_and_write_to_output(
            results=results, title=module_name.capitalize(), output_filename=module_name
        )


_ = get_backend_completeness_table()
