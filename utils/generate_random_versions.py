import random
from pathlib import Path

PANDAS_VERSION = [
    "1.0.5",
    "1.1.5",
    "1.2.5",
    "1.3.5",
    "1.4.4",
    "1.5.3",
    "2.0.3",
    "2.1.4",
    "2.2.2",
]
POLARS_VERSION = [
    "0.20.3",
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
    "11.0.0",
    "12.0.0",
    "12.0.1",
    "13.0.0",
    "14.0.0",
    "14.0.1",
    "14.0.2",
    "15.0.0",
    "15.0.1",
    "15.0.2",
    "16.0.0",
    "16.1.0",
]

pandas_version = random.choice(PANDAS_VERSION)
polars_version = random.choice(POLARS_VERSION)
pyarrow_version = random.choice(PYARROW_VERSION)

dependencies_content = (
    f"pandas=={pandas_version}\npolars=={polars_version}\npyarrow=={pyarrow_version}\n"
)

Path("random-requirements.txt").write_text(dependencies_content, encoding="UTF-8")

pyproject_path = Path("pyproject.toml")
pyproject_content = pyproject_path.read_text(encoding="UTF-8")

updated_pyproject_content = pyproject_content.replace(
    'filterwarnings = [\n  "error",\n]',
    "filterwarnings = [\n  \"error\",\n  'ignore:distutils Version classes are deprecated:DeprecationWarning',\n]",
)
pyproject_path.write_text(updated_pyproject_content, encoding="UTF-8")
