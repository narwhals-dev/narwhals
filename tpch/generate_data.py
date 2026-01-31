"""Generate database and answers via [DuckDB TPC-H Extension].

[DuckDB TPC-H Extension]: https://duckdb.org/docs/stable/core_extensions/tpch
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
from functools import cache
from typing import TYPE_CHECKING, Any

from tpch.classes import TableLogger
from tpch.constants import (
    DATABASE_TABLE_NAMES,
    GLOBS,
    LOGGER_NAME,
    QUERY_IDS,
    SCALE_FACTOR_DEFAULT,
    SCALE_FACTORS,
    _scale_factor_dir,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from pathlib import Path

    import polars as pl
    import pytest
    from duckdb import DuckDBPyConnection as Con, DuckDBPyRelation as Rel
    from typing_extensions import LiteralString

    from tpch.typing_ import Artifact, QueryID, ScaleFactor


logger = logging.getLogger(LOGGER_NAME)


# `mem_usage_scale = 2.705`
# `pl.Config(tbl_hide_column_data_types=True, tbl_hide_dataframe_shape=True)`
# https://duckdb.org/docs/stable/core_extensions/tpch#resource-usage-of-the-data-generator
TABLE_SCALE_FACTOR = """
┌───────┬────────────┬─────────────┐
│ sf    ┆ Disk       ┆ Memory (db) │
╞═══════╪════════════╪═════════════╡
│ 0.014 ┆    3.25 mb ┆    8.79 mb  │
│ 0.052 ┆   12.01 mb ┆   32.49 mb  │
│ 0.1   ┆   23.15 mb ┆   62.62 mb  │
│ 0.25  ┆   58.90 mb ┆  159.32 mb  │
│ 0.51  ┆  124.40 mb ┆  336.50 mb  │
│ 1.0   ┆  247.66 mb ┆  669.92 mb  │
│ 10.0  ┆    2.59 gb ┆    7.00 gb  │
│ 30.0  ┆    7.76 gb ┆   21.00 gb  │
└───────┴────────────┴─────────────┘
"""


# NOTE: Store queries here, add parameter names if needed
SQL_DBGEN = "CALL dbgen(sf={0})"
SQL_TPCH_ANSWER = "PRAGMA tpch({0})"
SQL_FROM = "FROM {0}"
SQL_SHOW_DB = """
SELECT
    "table": name,
    "schema": MAP(column_names, column_types)
FROM
    (SHOW ALL TABLES)
"""

FIX_ANSWERS: Mapping[QueryID, Callable[[pl.LazyFrame], pl.LazyFrame]] = {
    "q18": lambda df: df.rename({"sum(l_quantity)": "sum"}).cast({"sum": int}),
    "q22": lambda df: df.cast({"cntrycode": int}),
}
"""
DuckDB being weird, this is [correct] but [not this one].

[correct]: https://github.com/duckdb/duckdb/blob/47c227d7d8662586b0307d123c03b25c0db3d515/extension/tpch/dbgen/answers/sf0.01/q18.csv#L1
[not this one]: https://github.com/duckdb/duckdb/blob/47c227d7d8662586b0307d123c03b25c0db3d515/extension/tpch/dbgen/answers/sf100/q18.csv#L1
"""


def read_fmt_schema(fp: Path) -> str:
    import polars as pl

    schema = pl.read_parquet_schema(fp).items()
    return f"- {fp.name}\n" + "\n".join(f"  - {k:<20}: {v}" for k, v in schema)


@cache
def cast_map() -> dict[Any, Any]:
    import polars as pl
    import polars.selectors as cs

    return {cs.decimal(): float, cs.date(): pl.Datetime("ns")}


@dataclasses.dataclass(**({"kw_only": True} if sys.version_info >= (3, 10) else {}))
class TPCHGen:
    scale_factor: ScaleFactor
    refresh: bool = False
    debug: bool = False
    _con: Con = dataclasses.field(init=False, repr=False)

    @staticmethod
    def from_pytest(config: pytest.Config, /) -> TPCHGen:
        return TPCHGen(scale_factor=config.getoption("--scale-factor"))

    @staticmethod
    def from_argparse(parser: argparse.ArgumentParser, /) -> TPCHGen:
        return parser.parse_args(namespace=TPCHGen.__new__(TPCHGen))

    @property
    def scale_factor_dir(self) -> Path:
        return _scale_factor_dir(self.scale_factor)

    def glob(self, artifact: Artifact, /) -> Iterator[Path]:
        return self.scale_factor_dir.glob(GLOBS[artifact])

    def has_data(self) -> bool:
        both = next(self.glob("answers"), None) and next(self.glob("database"), None)
        return bool(both)

    def run(self) -> None:
        _configure_logger(debug=self.debug)
        if self.refresh:
            logger.info("Refreshing data for scale_factor=%s", self.scale_factor)
        elif self.has_data():
            logger.info("Data already exists for scale_factor=%s", self.scale_factor)
            self.show_schemas("database").show_schemas("answers")
            logger.info("To regenerate this scale_factor, use `--refresh`")
            return
        self.connect().load_extension().generate_database().write_database().write_answers()
        n_bytes = sum(e.stat().st_size for e in os.scandir(self.scale_factor_dir))
        total = TableLogger.format_size(n_bytes)
        logger.info("Finished with total file size: %s", total.strip())

    def connect(self) -> TPCHGen:
        import duckdb

        logger.info("Connecting to in-memory DuckDB database")
        self._con = duckdb.connect(database=":memory:")
        return self

    def sql(self, query: LiteralString) -> Rel:
        return self._con.sql(query)

    def load_extension(self) -> TPCHGen:
        logger.info("Installing DuckDB TPC-H Extension")
        self._con.install_extension("tpch")
        self._con.load_extension("tpch")
        return self

    def generate_database(self) -> TPCHGen:
        logger.info("Generating data for scale_factor=%s", self.scale_factor)
        self.sql(SQL_DBGEN.format(self.scale_factor))
        logger.info("Finished generating data.")
        if logger.isEnabledFor(logging.DEBUG):
            msg = str(self.sql(SQL_SHOW_DB))[:-1]
            logger.debug("DuckDB schemas (database):\n%s", msg)
        return self

    def write_database(self) -> TPCHGen:
        logger.info("Writing data to: %s", self.scale_factor_dir.as_posix())
        with TableLogger.database() as tbl_logger:
            for t in DATABASE_TABLE_NAMES:
                path = self.scale_factor_dir / f"{t}.parquet"
                to_polars(self.sql(SQL_FROM.format(t))).sink_parquet(path)
                tbl_logger.log_row(path)
        return self.show_schemas("database")

    def write_answers(self) -> TPCHGen:
        logger.info("Executing tpch queries for answers")
        with TableLogger.answers() as tbl_logger:
            for query_id in QUERY_IDS:
                query = SQL_TPCH_ANSWER.format(query_id.removeprefix("q"))
                lf = to_polars(self.sql(query))
                if fix := FIX_ANSWERS.get(query_id):
                    lf = fix(lf)
                path = self.scale_factor_dir / f"result_{query_id}.parquet"
                lf.sink_parquet(path)
                tbl_logger.log_row(path)
        return self.show_schemas("answers")

    def show_schemas(self, artifact: Artifact, /) -> TPCHGen:
        if logger.isEnabledFor(logging.DEBUG):
            if paths := sorted(self.glob(artifact)):
                msg = "\n".join(read_fmt_schema(fp) for fp in paths)
                logger.debug("Parquet schemas (%s):\n%s", artifact, msg)
            else:
                msg = f"Found no matching paths for {artifact!r} in {self.scale_factor_dir.as_posix()}"
                raise NotImplementedError(msg)
        return self


def to_polars(rel: Rel) -> pl.LazyFrame:
    return rel.pl(lazy=True).cast(cast_map())


def _configure_logger(
    *,
    debug: bool,
    fmt: str = "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt: str = "%H:%M:%S",
) -> None:
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    output = logging.StreamHandler()
    output.setFormatter(logging.Formatter(fmt, datefmt))
    logger.addHandler(output)


class HelpFormatter(
    argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    def start_section(self, heading: str | None) -> None:
        super().start_section(_title.capitalize() if (_title := heading) else heading)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=HelpFormatter,
        description="Generate the data required to run TPCH queries.\n\nUsage: %(prog)s [OPTIONS]",
        usage=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-sf",
        "--scale-factor",
        default=SCALE_FACTOR_DEFAULT,
        metavar="",
        help=f"Scale the database by this factor (default: %(default)s)\n{TABLE_SCALE_FACTOR}",
        choices=SCALE_FACTORS,
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable more detailed logging"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-run data generation, regardless of whether `--scale-factor` is already on disk",
    )
    TPCHGen.from_argparse(parser).run()
