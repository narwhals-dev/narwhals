from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PANDAS_AND_NUMPY_VERSION = (
    ("1.3.5", "1.21.6"),
    ("1.4.4", "1.22.4"),
    ("1.5.3", "1.23.5"),
    ("2.0.3", "1.24.4"),
    ("2.1.4", "1.25.2"),
    ("2.2.2", "1.26.4"),
)
POLARS_VERSION = (
    "0.20.4",
    "0.20.5",
    "0.20.6",
    "0.20.7",
    "0.20.8",
    "0.20.9",
    "0.20.10",
    "0.20.13",
    "0.20.14",
    "0.20.15",
    "0.20.16",
    "0.20.17",
    "0.20.18",
    "0.20.19",
    "0.20.21",
    "0.20.22",
    "0.20.23",
    "0.20.25",
    "0.20.26",
    "0.20.30",
    "0.20.31",
    "1.0.0",
    "1.1.0",
)
PYARROW_VERSION = (
    "13.0.0",
    "14.0.0",
    "14.0.1",
    "14.0.2",
    "15.0.0",
    "15.0.1",
    "15.0.2",
    "16.0.0",
    "16.1.0",
    "17.0.0",
    "18.0.0",
    "18.1.0",
)


def all_combos() -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (pd, np, pl, pa)
        for pd, np in PANDAS_AND_NUMPY_VERSION
        for pl in POLARS_VERSION
        for pa in PYARROW_VERSION
    )


def sample_distinct(n: int) -> list[dict[str, str]]:
    pool = all_combos()
    if n > len(pool):
        msg = f"Requested {n} distinct combos but only {len(pool)} exist."
        raise ValueError(msg)
    picks = random.sample(pool, n)
    return [
        {"pandas": pd, "numpy": np, "polars": pl, "pyarrow": pa}
        for pd, np, pl, pa in picks
    ]


def to_requirements(combo: dict[str, str]) -> str:
    return (
        f"numpy=={combo['numpy']}\n"
        f"pandas=={combo['pandas']}\n"
        f"polars=={combo['polars']}\n"
        f"pyarrow=={combo['pyarrow']}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=1)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help=(
            "Path to write to. A `.json` extension writes a JSON array of combos; "
            "any other extension writes a requirements.txt-style file (requires n=1)."
        ),
    )
    args = parser.parse_args()

    combos = sample_distinct(args.num)

    if args.output.suffix == ".json":
        args.output.write_text(json.dumps(combos), "utf-8")
        return

    if args.num != 1:
        msg = f"Non-JSON output ({args.output.suffix or 'no extension'}) requires --num=1"
        raise ValueError(msg)
    args.output.write_text(to_requirements(combos[0]), "utf-8")


if __name__ == "__main__":
    main()
