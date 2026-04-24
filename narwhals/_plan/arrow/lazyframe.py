from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import pyarrow as pa

from narwhals._plan._version import into_version
from narwhals._plan.common import todo
from narwhals._plan.compliant.lazyframe import CompliantLazyFrame
from narwhals._utils import Implementation, Version, unstable

if TYPE_CHECKING:
    import polars as pl
    from typing_extensions import Self

    from narwhals.schema import Schema

MAIN = Version.MAIN


@unstable
class ArrowLazyFrame(CompliantLazyFrame[pa.Table]):
    """Experiment, expect lots of changes.

    Representing with `pa.Table` is *an option*, but may only need for a subset of operations.

    Other choices:
    - `pyarrow.acero.Declaration`
      - Sits somewhere between `LogicalPlan` and `ResolvedPlan`
      - Doesn't require the schema at every step
      - But many nodes need to know the column names
        - No concept of `with_columns`, so things to preserve must be tracked
    - `pyarrow.dataset.Dataset`
      - Has builder methods that internally add `Declaration`s or wrap `Scanner` methods
    - `pyarrow.dataset.Scanner`
      - Collection operations
      - Has both `dataset_schema` and (uniquely) `projected_schema` properties
    """

    __slots__ = ("_input_schema", "_native")
    implementation: ClassVar = Implementation.PYARROW
    version: ClassVar[Version] = Version.MAIN

    _native: pa.Table
    _input_schema: Schema | None

    from_arrow = todo()
    from_pandas = todo()
    from_narwhals = todo()

    @classmethod
    def from_native(cls, native: pa.Table, /, version: Version = MAIN) -> Self:  # noqa: ARG003
        obj = cls.__new__(cls)
        obj._native = native
        return obj

    @classmethod
    def from_polars(cls, frame: pl.DataFrame, /, version: Version = MAIN) -> Self:
        return cls.from_native(frame.to_arrow(), version)

    @property
    def input_schema(self) -> Schema:
        if self._input_schema is None:
            self._input_schema = into_version(self).schema.from_arrow(self.native.schema)
        return self._input_schema

    @property
    def native(self) -> pa.Table:
        return self._native

    collect_schema = todo()
    collect_arrow = todo()
    collect_polars = todo()
    collect_pandas = todo()

    from_compliant = from_narwhals
