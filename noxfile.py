from __future__ import annotations

from typing import TYPE_CHECKING

import nox

if TYPE_CHECKING:
    from nox.sessions import Session

nox.options.default_venv_backend = "uv"
nox.options.reuse_venv = True

PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11", "3.12"]


def run_common(session: Session, coverage_threshold: float) -> None:
    session.install("-e.", "-r", "requirements-dev.txt")

    if session.python != "3.8":
        session.install("ibis-framework[duckdb]>=6.0.0")

    session.run(
        "pytest",
        "tests",
        "--cov=narwhals",
        "--cov=tests",
        f"--cov-fail-under={coverage_threshold}",
        "--runslow",
    )
    session.run("pytest", "narwhals", "--doctest-modules")


@nox.session(python=PYTHON_VERSIONS)  # type: ignore[misc]
def pytest_coverage(session: Session) -> None:
    if session.python == "3.8":
        coverage_threshold = 85
    else:
        coverage_threshold = 100
        session.install("modin[dask]")

    run_common(session, coverage_threshold)


@nox.session(python=PYTHON_VERSIONS[0])  # type: ignore[misc]
@nox.parametrize("pandas_version", ["0.25.3", "1.1.5"])  # type: ignore[misc]
def min_and_old_versions(session: Session, pandas_version: str) -> None:
    session.install(
        f"pandas=={pandas_version}",
        "polars==0.20.3",
        "numpy==1.17.5",
        "pyarrow==11.0.0",
        "scipy==1.5.0",
        "scikit-learn==1.1.0",
        "tzdata",
    )
    run_common(session, coverage_threshold=50)


@nox.session(python=PYTHON_VERSIONS[-1])  # type: ignore[misc]
def nightly_versions(session: Session) -> None:
    session.install("polars")

    session.install(  # pandas nightly
        "--pre",
        "--extra-index-url",
        "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple",
        "pandas",
    )

    session.install(  # numpy nightly
        "--pre",
        "--extra-index-url",
        "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple",
        "numpy",
    )

    session.run("uv", "pip", "install", "pip")
    session.run(  # dask nightly
        "pip",
        "install",
        "git+https://github.com/dask/distributed",
        "git+https://github.com/dask/dask",
        "git+https://github.com/dask/dask-expr",
    )
    run_common(session, coverage_threshold=50)
