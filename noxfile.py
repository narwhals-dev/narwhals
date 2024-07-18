import nox
from nox.sessions import Session

nox.options.default_venv_backend = "uv"
nox.options.reuse_venv = True

PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11", "3.12"]


def run_common(session: Session, coverage_threshold: float) -> None:
    session.install("-e.", "-r", "requirements-dev.txt")

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
    coverage_threshold = 90 if session.python == "3.8" else 100

    session.install("modin[dask]")

    run_common(session, coverage_threshold)


@nox.session(python=PYTHON_VERSIONS[0])  # type: ignore[misc]
def minimum_versions(session: Session) -> None:
    session.install(
        "pandas==0.25.3",
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
    session.install("modin[dask]", "polars")
    session.install(
        "--pre",
        "--extra-index-url",
        "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple",
        "pandas",
    )

    run_common(session, coverage_threshold=50)
