from __future__ import annotations

from typing import TYPE_CHECKING, Any

from narwhals._compliant import CompliantNamespace
from narwhals._utils import Implementation, not_implemented
from test_plugin.dataframe import DictDataFrame, DictFrame, DictLazyFrame

if TYPE_CHECKING:
    from narwhals.typing import NormalizedPath
    from narwhals.utils import Version
    from test_plugin.series import DictSeries


class DictNamespace(CompliantNamespace[DictLazyFrame, Any]):
    def __init__(self, *, version: Version) -> None:
        self._version = version

    def from_native(self, native_object: DictFrame) -> DictLazyFrame:
        return DictLazyFrame(native_object, version=self._version)

    @property
    def _dataframe(self) -> type[DictDataFrame]:
        return DictDataFrame

    @property
    def _series(self) -> type[DictSeries]:
        from test_plugin.series import DictSeries

        return DictSeries

    # IO methods below follow the namespace contract used by `narwhals.functions`
    # (see "IO functions: the namespace contract" in `docs/extending.md`).
    # `read_*` are deliberately left unimplemented: `test_plugin` wraps dicts
    # lazily, so it only supports `scan_*`.

    def scan_csv(
        self, source: NormalizedPath, *, separator: str = ",", **kwds: Any
    ) -> DictLazyFrame:
        import csv
        from pathlib import Path

        with Path(source).open(newline="", encoding="utf-8") as file:
            header, *rows = list(csv.reader(file, delimiter=separator))
        data = {name: [row[index] for row in rows] for index, name in enumerate(header)}
        return DictLazyFrame(data, version=self._version)

    def scan_parquet(self, source: NormalizedPath, **kwds: Any) -> DictLazyFrame:
        import pyarrow.parquet as pq

        data: DictFrame = pq.read_table(source, **kwds).to_pydict()
        return DictLazyFrame(data, version=self._version)

    # NOTE: `not_implemented.__get__` reads `instance._implementation` to build its
    # error message, so `_implementation` itself must be a real value.
    _implementation = Implementation.UNKNOWN

    is_native: Any = not_implemented()
    _expr: Any = not_implemented()
    corr: Any = not_implemented()
    cov: Any = not_implemented()
    len: Any = not_implemented()
    list: Any = not_implemented()
    lit: Any = not_implemented()
    all_horizontal: Any = not_implemented()
    any_horizontal: Any = not_implemented()
    sum_horizontal: Any = not_implemented()
    mean_horizontal: Any = not_implemented()
    min_horizontal: Any = not_implemented()
    max_horizontal: Any = not_implemented()
    concat: Any = not_implemented()
    when: Any = not_implemented()
    concat_str: Any = not_implemented()
    selectors: Any = not_implemented()
    coalesce: Any = not_implemented()
    struct: Any = not_implemented()
