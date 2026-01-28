"""Generate database and answers via [DuckDB TPC-H Extension].

[DuckDB TPC-H Extension]: https://duckdb.org/docs/stable/core_extensions/tpch
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import logging
import os
from functools import cache
from typing import TYPE_CHECKING, Any

import polars as pl
import polars.selectors as cs

from tpch.classes import TableLogger
from tpch.constants import (
    DATA_DIR,
    DATABASE_TABLE_NAMES,
    GLOBS,
    LOGGER_NAME,
    QUERY_IDS,
    get_scale_factor_dir,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from duckdb import DuckDBPyConnection as Con

    from tpch.typing_ import Artifact, BuiltinScaleFactor, QueryID


logger = logging.getLogger(LOGGER_NAME)


def read_fmt_schema(fp: Path) -> str:
    schema = pl.read_parquet_schema(fp).items()
    return f"- {fp.name}\n" + "\n".join(f"  - {k:<20}: {v}" for k, v in schema)


def show_schemas(artifact: Artifact, scale_factor: float, /) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    pattern = GLOBS[artifact]
    sf_dir = get_scale_factor_dir(scale_factor)
    paths = sorted(sf_dir.glob(pattern))
    if not paths:
        msg = f"Found no matching paths for {pattern!r} in {sf_dir.as_posix()}"
        raise NotImplementedError(msg)
    msg = "\n".join(read_fmt_schema(fp) for fp in paths)
    logger.debug("Schemas (%s):\n%s", artifact, msg)


TABLE_SCALE_FACTOR = """
┌──────────────┬───────────────┐
│ Scale factor ┆ Database (MB) │
╞══════════════╪═══════════════╡
│ 0.1          ┆ 25            │
│ 1.0          ┆ 250           │
│ 3.0          ┆ 754           │
│ 100.0        ┆ 26624         │
└──────────────┴───────────────┘
"""


SF_BUILTIN_STR: Mapping[float, BuiltinScaleFactor] = {
    0.01: "0.01",
    0.1: "0.1",
    1.0: "1.0",
}


# NOTE: Store queries here, add parameter names if needed
SQL_DBGEN = "CALL dbgen(sf={0})"
SQL_TPCH_ANSWER = "PRAGMA tpch({0})"
SQL_TPCH_ANSWERS = """
    SELECT query_nr, answer
    FROM tpch_answers()
    WHERE scale_factor={0}
"""
SQL_FROM = "FROM {0}"


FIX_ANSWERS: Mapping[QueryID, Callable[[pl.DataFrame], pl.DataFrame]] = {
    "q18": lambda df: df.rename({"sum(l_quantity)": "sum"}).cast({"sum": pl.Int64()}),
    "q22": lambda df: df.cast({"cntrycode": pl.Int64()}),
}
"""
DuckDB being weird, this is [correct] but [not this one].

[correct]: https://github.com/duckdb/duckdb/blob/47c227d7d8662586b0307d123c03b25c0db3d515/extension/tpch/dbgen/answers/sf0.01/q18.csv#L1
[not this one]: https://github.com/duckdb/duckdb/blob/47c227d7d8662586b0307d123c03b25c0db3d515/extension/tpch/dbgen/answers/sf100/q18.csv#L1
"""


@cache
def cast_map() -> dict[Any, Any]:
    casts: dict[Any, Any] = {
        cs.decimal(): float,
        cs.date() | cs.by_name("o_orderdate", require_all=False): pl.Datetime("ns"),
    }
    return casts


def connect() -> Con:
    import duckdb

    logger.info("Connecting to in-memory DuckDB database")
    return duckdb.connect(database=":memory:")


def load_tpch_extension(con: Con) -> Con:
    logger.info("Installing DuckDB TPC-H Extension")
    con.install_extension("tpch")
    con.load_extension("tpch")
    return con


def generate_tpch_database(con: Con, scale_factor: float) -> Con:
    logger.info("Generating data with scale_factor=%s", scale_factor)
    con.sql(SQL_DBGEN.format(scale_factor))
    logger.info("Finished generating data.")
    return con


def write_tpch_database(con: Con, scale_factor: float) -> Con:
    sf_dir = get_scale_factor_dir(scale_factor)
    sf_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing data to: %s", sf_dir.as_posix())
    with TableLogger.database() as tbl_logger:
        for t in DATABASE_TABLE_NAMES:
            df = con.sql(SQL_FROM.format(t)).pl().cast(cast_map())
            path = sf_dir / f"{t}.parquet"
            df.write_parquet(path)
            tbl_logger.log_row(path.name, path.stat().st_size)
    show_schemas("database", scale_factor)
    return con


def _answers_any(con: Con, scale_factor: float) -> Con:
    sf_dir = get_scale_factor_dir(scale_factor)
    logger.info("Executing tpch queries for answers")
    with TableLogger.answers() as tbl_logger:
        for query_id in QUERY_IDS:
            query = SQL_TPCH_ANSWER.format(query_id.removeprefix("q"))
            df = con.sql(query).pl().cast(cast_map())
            if fix := FIX_ANSWERS.get(query_id):
                df = fix(df)
            path = sf_dir / f"result_{query_id}.parquet"
            df.write_parquet(path)
            tbl_logger.log_row(path.name, path.stat().st_size)
    return con


def _answers_builtin(con: Con, scale: BuiltinScaleFactor, scale_factor: float) -> Con:
    sf_dir = get_scale_factor_dir(scale_factor)
    logger.info("Fastpath for builtin tpch_answers()")
    results = con.sql(SQL_TPCH_ANSWERS.format(scale))
    with TableLogger.answers() as tbl_logger:
        while row := results.fetchmany(1):
            query_nr, answer = row[0]
            source = io.BytesIO(answer.encode("utf-8"))
            df = pl.read_csv(source, separator="|", try_parse_dates=True).cast(cast_map())
            path = sf_dir / f"result_q{query_nr}.parquet"
            df.write_parquet(path)
            tbl_logger.log_row(path.name, path.stat().st_size)
    return con


def write_tpch_answers(con: Con, scale_factor: float) -> Con:
    logger.info("Getting answers")
    con = (
        _answers_builtin(con, scale, scale_factor)
        if (scale := SF_BUILTIN_STR.get(scale_factor))
        else _answers_any(con, scale_factor)
    )
    show_schemas("answers", scale_factor)
    return con


def write_metadata(scale_factor: float) -> None:
    """Write metadata.json to the scale factor directory with creation timestamp."""
    sf_dir = get_scale_factor_dir(scale_factor)
    metadata_path = sf_dir / "_metadata.json"
    logger.info("Writing metadata to: %s", metadata_path.as_posix())

    metadata = {"created_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    metadata_path.write_text(json.dumps(metadata, indent=2))


def scale_factor_exists(scale_factor: float) -> bool:
    """Check if data for a scale factor exists by checking if its directory exists."""
    sf_dir = get_scale_factor_dir(scale_factor)
    return sf_dir.exists()


def main(*, scale_factor: float = 0.1, refresh: bool = False) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if refresh:
        logger.info("Refreshing data for scale_factor=%s", scale_factor)
    elif scale_factor_exists(scale_factor):
        logger.info("Data already exists for scale_factor=%s", scale_factor)
        show_schemas("database", scale_factor)
        show_schemas("answers", scale_factor)
        logger.info("To regenerate this scale_factor, use `--refresh`")
        return

    logger.info("Generating data for scale_factor=%s", scale_factor)
    con = connect()
    load_tpch_extension(con)
    generate_tpch_database(con, scale_factor)
    write_tpch_database(con, scale_factor)
    write_tpch_answers(con, scale_factor)
    write_metadata(scale_factor)
    total = TableLogger.format_size(sum(e.stat().st_size for e in os.scandir(DATA_DIR)))
    logger.info("Finished with total file size: %s", total.strip())


def _configure_logger(
    *,
    debug: bool,
    fmt: str = "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    output = logging.StreamHandler()
    output.setFormatter(logging.Formatter(fmt, datefmt))
    logger.addHandler(output)


class HelpFormatter(
    argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    def start_section(self, heading: str | None) -> None:
        title = _title.capitalize() if (_title := heading) else heading
        super().start_section(title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=HelpFormatter,
        description="Generate the data required to run TPCH queries.\n\nUsage: %(prog)s [OPTIONS]",
        usage=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-sf",
        "--scale-factor",
        default="0.1",
        metavar="",
        help=f"Scale the database by this factor (default: %(default)s)\n{TABLE_SCALE_FACTOR}",
        type=float,
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable more detailed logging"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-run data generation, regardless of whether `--scale-factor` is already on disk",
    )
    args = parser.parse_args()
    _configure_logger(debug=args.debug)
    main(scale_factor=args.scale_factor, refresh=args.refresh)
