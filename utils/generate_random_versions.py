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


def sample_distinct(n: int) -> list[dict[str, str]]:
    """Return `n` combos where no version of any single library is reused."""
    n_max = min(len(PANDAS_AND_NUMPY_VERSION), len(POLARS_VERSION), len(PYARROW_VERSION))
    if n > n_max:
        msg = (
            f"Requested {n} combos but at most {n_max} are possible "
            "without reusing a version of any single library."
        )
        raise ValueError(msg)
    pandas_numpy = random.sample(PANDAS_AND_NUMPY_VERSION, n)
    polars = random.sample(POLARS_VERSION, n)
    pyarrow = random.sample(PYARROW_VERSION, n)
    return [
        {"pandas": pd, "numpy": np, "polars": pl, "pyarrow": pa}
        for (pd, np), pl, pa in zip(pandas_numpy, polars, pyarrow, strict=True)
    ]


def to_requirements(combo: dict[str, str]) -> str:
    """Render a single combo as the contents of a requirements.txt file."""
    return (
        f"numpy=={combo['numpy']}\n"
        f"pandas=={combo['pandas']}\n"
        f"polars=={combo['polars']}\n"
        f"pyarrow=={combo['pyarrow']}\n"
    )


def main() -> None:
    """Generate version combos and write them to the requested output path."""
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

    num: int = args.num
    output: Path = args.output

    combos = sample_distinct(n=num)

    if output.suffix == ".json":
        output.write_text(json.dumps(combos), "utf-8")
        return

    if num != 1:
        msg = f"Non-JSON output ({output.suffix or 'no extension'}) requires --num=1"
        raise ValueError(msg)

    output.write_text(to_requirements(combos[0]), "utf-8")


if __name__ == "__main__":
    main()
