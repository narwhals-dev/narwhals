from __future__ import annotations

import random
from pathlib import Path

PANDAS_AND_NUMPY_VERSION = [
    ("1.3.5", "1.21.6"),
    ("1.4.4", "1.22.4"),
    ("1.5.3", "1.23.5"),
    ("2.0.3", "1.24.4"),
    ("2.1.4", "1.25.2"),
    ("2.2.2", "1.26.4"),
]
POLARS_VERSION = [
    "1.0.0",
    "1.1.0",
    "1.2.1",
    "1.3.0",
    "1.4.1",
    "1.5.0",
    "1.6.0",
    "1.7.1",
    "1.8.2",
    "1.9.0",
    "1.10.0",
    "1.11.0",
    "1.12.0",
    "1.14.0",
    "1.15.0",
    "1.16.0",
    "1.17.0",
    "1.17.1",
    "1.18.0",
    "1.19.0",
    "1.20.0",
    "1.21.0",
    "1.22.0",
    "1.23.0",
    "1.24.0",
    "1.25.2",
    "1.26.0",
    "1.27.1",
    "1.28.1",
    "1.29.0",
    "1.30.0",
    "1.31.0",
    "1.32.0",
    "1.32.2",
    "1.32.3",
    "1.33.0",
    "1.33.1",
    "1.34.0",
    "1.35.1",
    "1.35.2",
    "1.36.1",
    "1.37.0",
    "1.37.1",
    "1.38.1",
    "1.39.0",
    "1.39.3",
    "1.40.0",
    "1.40.1",
    "1.41.1",
    "1.41.2",
    "1.42.0",
    "1.42.1",
    "1.43.2",
    "1.44.1",
    "1.44.2",
]
PYARROW_VERSION = [
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
]

pandas_version, numpy_version = random.choice(PANDAS_AND_NUMPY_VERSION)
polars_version = random.choice(POLARS_VERSION)
pyarrow_version = random.choice(PYARROW_VERSION)

reqs = f"pandas=={pandas_version}\nnumpy=={numpy_version}\npolars=={polars_version}\npyarrow=={pyarrow_version}\n"
Path("random-requirements.txt").write_text(reqs, "utf-8")
