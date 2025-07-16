from __future__ import annotations

import random
from pathlib import Path

PANDAS_AND_NUMPY_VERSION = [
    ("1.1.5", "1.19.5"),
    ("1.2.5", "1.20.3"),
    ("1.3.5", "1.21.6"),
    ("1.4.4", "1.22.4"),
    ("1.5.3", "1.23.5"),
    ("2.0.3", "1.24.4"),
    ("2.1.4", "1.25.2"),
    ("2.2.2", "1.26.4"),
]
POLARS_VERSION = [
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
old_warnings = 'filterwarnings = [\n  "error",\n]'
new_warnings = "filterwarnings = [\n  \"error\",\n  'ignore:distutils Version classes are deprecated:DeprecationWarning',\n]"
pyproject = Path("pyproject.toml")
content = pyproject.read_text("utf-8").replace(old_warnings, new_warnings)
pyproject.write_text(content, "utf-8")
