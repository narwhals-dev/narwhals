from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from narwhals.utils import Version
    from test_plugin.dataframe import DictFrame
    from test_plugin.namespace import DictNamespace


def __narwhals_namespace__(version: Version) -> DictNamespace:  # noqa: N807
    from test_plugin.namespace import DictNamespace

    return DictNamespace(version=version)


def is_native(native_object: object) -> TypeIs[DictFrame]:
    return isinstance(native_object, dict)


NATIVE_PACKAGE = "builtins"


# The functions below are the IO extension hooks used by `narwhals.functions`
# when `backend` resolves to this plugin (https://github.com/narwhals-dev/narwhals/issues/3713).
# Eager constructors (`from_dict`, `new_series`, `Series.from_iterable`, ...) instead go
# through `DictNamespace._dataframe` / `DictNamespace._series`.


def scan_csv(source: str, separator: str = ",", **kwargs: Any) -> DictFrame:  # noqa: ARG001
    import csv
    from pathlib import Path

    with Path(source).open(newline="", encoding="utf-8") as file:
        header, *rows = list(csv.reader(file, delimiter=separator))
    return {name: [row[index] for row in rows] for index, name in enumerate(header)}


def scan_parquet(source: str, **kwargs: Any) -> DictFrame:
    import pyarrow.parquet as pq

    result: DictFrame = pq.read_table(source, **kwargs).to_pydict()
    return result


# `read_*` returns the same native object; narwhals then enforces eagerness downstream.
read_csv = scan_csv
read_parquet = scan_parquet
