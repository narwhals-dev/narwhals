from __future__ import annotations

from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

import nox

if TYPE_CHECKING:
    from nox.sessions import Session

nox.options.default_venv_backend = "uv"
nox.options.reuse_venv = "never"

PYTHON_VERSIONS = {
    "pytest": ["3.8", "3.10", "3.11", "3.12", "3.13"],
    "random": ["3.9"],
    "minimum": "3.8",
    "pretty_old": "3.8",
    "not_so_old": "3.10",
    "nightly": "3.13",
}


@nox.session(python=PYTHON_VERSIONS["pytest"])
def pytest_coverage(session: Session) -> None:
    pytest_cmd = [
        "pytest",
        "tests",
        "--cov=narwhals",
        "--cov=tests",
    ]

    if session.python == "3.8":
        session.install(
            "-e", ".[pandas,polars,pyarrow]", "backports.zoneinfo", "--group", "tests"
        )
        session.run(
            *pytest_cmd,
            "--cov-fail-under=80",
            "--constructors",
            "pandas,pyarrow,polars[eager],polars[lazy]",
        )

    elif session.python in {"3.10", "3.12"}:
        session.install(
            "-e", ".[dask,modin]", "--group", "core-tests", "--group", "extra"
        )
        session.run(
            *pytest_cmd,
            "--cov-fail-under=95",
            "--runslow",
            "--constructors",
            "pandas,pandas[nullable],pandas[pyarrow],pyarrow,modin[pyarrow],polars[eager],polars[lazy],dask,duckdb,sqlframe",
        )

    elif session.python in {"3.11", "3.13"}:
        session.install(
            "-e", ".[modin, dask]", "--group", "core-tests", "--group", "extra"
        )
        session.install("-U", "--pre", "duckdb")
        if session.python != "3.13":
            with NamedTemporaryFile() as f:
                f.write(b"setuptools<78\n")
                session.install("-b", f.name, "-e", ".[pyspark]")

        if session.python == "3.11":
            session.install("-e", ".[ibis]")

        session.run(
            *pytest_cmd,
            "--cov-fail-under=100",
            "--runslow",
            "--all-cpu-constructors",
            env={
                "NARWHALS_POLARS_NEW_STREAMING": str(session.python == "3.11"),
            },
        )

    if session.python == PYTHON_VERSIONS["pytest"][-1]:
        session.run("pytest", "narwhals/", "--doctest-modules")


@nox.session(python=PYTHON_VERSIONS["minimum"])
def minimum(session: Session) -> None:
    session.install(
        "pandas==0.25.3",
        "polars==0.20.3",
        "numpy==1.17.5",
        "pyarrow==11.0.0",
        "pyarrow-stubs<17",
        "scipy==1.5.0",
        "scikit-learn==1.1.0",
        "duckdb==1.0",
        "tzdata",
        "backports.zoneinfo",
    )
    session.install("-e", ".", "--group", "tests")
    session.run(
        "pytest",
        "tests",
        "--cov=narwhals",
        "--cov=tests",
        "--cov-fail-under=50",
        "--runslow",
        "--constructors=pandas,pyarrow,polars[eager],polars[lazy]",
    )


@nox.session(python=PYTHON_VERSIONS["pretty_old"])
def pretty_old(session: Session) -> None:
    session.install(
        "pandas==1.1.5",
        "polars==0.20.3",
        "numpy==1.17.5",
        "pyarrow==11.0.0",
        "pyarrow-stubs<17",
        "scipy==1.5.0",
        "scikit-learn==1.1.0",
        "duckdb==1.0",
        "tzdata",
        "backports.zoneinfo",
    )
    session.install("-e", ".", "--group", "tests")
    session.run(
        "pytest",
        "tests",
        "--cov=narwhals",
        "--cov=tests",
        "--cov-fail-under=50",
        "--runslow",
        "--constructors=pandas,pyarrow,polars[eager],polars[lazy]",
    )


@nox.session(python=PYTHON_VERSIONS["not_so_old"])
def not_so_old(session: Session) -> None:
    session.install(
        "pandas==2.0.3",
        "polars==0.20.8",
        "numpy==1.24.4",
        "pyarrow==15.0.0",
        "pyarrow-stubs<17",
        "scipy==1.8.0",
        "scikit-learn==1.3.0",
        "duckdb==1.0",
        "dask[dataframe]==2024.10 ",
        "tzdata",
    )
    session.install("-e", ".", "--group", "tests")
    session.run(
        "pytest",
        "tests",
        "--cov=narwhals",
        "--cov=tests",
        "--cov-fail-under=50",
        "--runslow",
        "--constructors=pandas,pyarrow,polars[eager],polars[lazy]",
    )


@nox.session(python=PYTHON_VERSIONS["nightly"])
def nightly_versions(session: Session) -> None:
    session.install("-e", ".", "--group", "tests")

    session.install("--pre", "polars")

    session.run("pip", "uninstall", "pandas", "--yes")
    session.install(  # pandas nightly
        "--pre",
        "--extra-index-url",
        "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple",
        "pandas",
    )

    session.run("pip", "uninstall", "pyarrow", "--yes")
    session.install(  # pyarrow nightly
        "--extra-index-url", "https://pypi.fury.io/arrow-nightlies/", "--pre", "pyarrow"
    )

    session.run("pip", "uninstall", "numpy", "--yes")
    session.install(  # numpy nightly
        "--pre",
        "--extra-index-url",
        "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple",
        "numpy",
    )

    session.run(  # dask nightly
        "pip",
        "install",
        "git+https://github.com/dask/distributed",
        "git+https://github.com/dask/dask",
    )

    session.install("-U", "--pre", "duckdb")  # duckdb nightly

    session.run(
        "pytest",
        "tests",
        "--cov=narwhals",
        "--cov=tests",
        "--cov-fail-under=50",
        "--runslow",
        "--constructors=pandas,pandas[nullable],pandas[pyarrow],pyarrow,polars[eager],polars[lazy],dask,duckdb",
    )


@nox.session(python=PYTHON_VERSIONS["random"])
def random(session: Session) -> None:
    from utils.generate_random_versions import requirements

    with NamedTemporaryFile("w", suffix=".txt") as fd:
        fd.write(requirements)
        fd.flush()

        session.install("-r", fd.name)
        session.install("-e", ".", "--group", "tests")

    session.run(
        "pytest",
        "tests",
        "--cov=narwhals",
        "--cov=tests",
        "--cov-fail-under=80",
        "--constructors=pandas,pyarrow,polars[eager],polars[lazy]",
    )
