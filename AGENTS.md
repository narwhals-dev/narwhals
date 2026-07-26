# AGENTS.md

Narwhals is an extremely lightweight, zero-dependency compatibility layer between Python dataframe libraries.

It lets library authors write dataframe-agnostic code once using a subset of the Polars API,
and have it work across pandas, Polars, PyArrow, cuDF, Modin, Dask, DuckDB, PySpark, Ibis, and SQLFrame,
without depending on any of them.

The primary audience is **library maintainers**, not end users.
Because of that, stability and backwards compatibility are taken extremely seriously.

## Read the docs first

Almost everything an agent needs is already documented. Read the relevant page instead of inferring
from the source, and update the page when you change the behaviour it describes.

| Topic | Read |
| --- | --- |
| Internal architecture: expressions, nodes, expression metadata, broadcasting, `over` push-down, group-by | [docs/how_it_works.md](docs/how_it_works.md) |
| Contributor workflow: env setup, test invocations, backend-specific rules, docstring style, PR conventions | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Stable API guarantees and the `main` vs `stable.v1` / `stable.v2` diff | [docs/backcompat.md](docs/backcompat.md) |
| Adding a backend: compliant protocols, plugin entry points, the IO namespace contract | [docs/extending.md](docs/extending.md) |
| Row-order semantics: what `DataFrame` guarantees, what `LazyFrame` doesn't, `over(order_by=...)` | [docs/concepts/order_dependence.md](docs/concepts/order_dependence.md) |
| Null vs NaN: which methods exist for which, and what pandas muddies | [docs/concepts/null_handling.md](docs/concepts/null_handling.md) |
| Why the pandas `group_by` `UserWarning` exists and how to avoid triggering it | [docs/concepts/improve_group_by_operation.md](docs/concepts/improve_group_by_operation.md) |
| Boolean semantics, the pandas index, non-string column names | [docs/concepts/](docs/concepts/) |
| Which methods each backend implements | [docs/api-completeness/](docs/api-completeness/) (generated, do not hand-edit) |
| Public API surface | [docs/api-reference/](docs/api-reference/) (member lists are validated by CI) |
| `narwhals.sql`: generating SQL from Narwhals expressions | [docs/generating_sql.md](docs/generating_sql.md) |
| Security reporting and release-permission policy | [docs/security.md](docs/security.md) |

The one-sentence summary of [docs/how_it_works.md](docs/how_it_works.md), worth internalising before
touching anything in `_pandas_like/`, `_arrow/`, or `_compliant/`:

> An expression is a function from a DataFrame to a sequence of Series.

## Layered design

1. **Public API** ([src/narwhals/dataframe.py](src/narwhals/dataframe.py),
   [series.py](src/narwhals/series.py), [expr.py](src/narwhals/expr.py), ...): the user-facing
   Polars-like API. Thin wrappers that build `ExprNode`s and dispatch to compliant backends.
2. **Compliant wrappers** (`src/narwhals/_pandas_like/`, `_arrow/`, `_polars/`, `_duckdb/`,
   `_spark_like/`, `_dask/`, `_ibis/`): each backend implements Narwhals-compliant DataFrames,
   Series, Exprs, and Namespaces that translate the Polars-like API to native calls. Shared
   protocols and base classes live in `src/narwhals/_compliant/`.
3. **Native libraries** (pandas, Polars, PyArrow, ...): the actual computation engines, never
   directly depended on.

## Source layout

```
src/narwhals/
  _pandas_like/    # Compliant layer for pandas, Modin, cuDF (fireducks is silently allowed here
                   # as a pandas drop-in; see `IMPORT_HOOKS` in dependencies.py)
  _arrow/          # Compliant layer for PyArrow
  _polars/         # Compliant layer for Polars
  _duckdb/         # Compliant layer for DuckDB
  _spark_like/     # Compliant layer for PySpark, Spark Connect, SQLFrame
  _dask/           # Compliant layer for Dask
  _ibis/           # Compliant layer for Ibis
  _interchange/    # Interchange protocol support
  _sql/            # Shared SQL generation utilities
  _compliant/      # Protocols and base classes shared across backends
  stable/          # Frozen stable API namespaces (v1, v2)
  testing/         # Public testing utilities (e.g. asserts)
  _expression_parsing.py  # ExprNode, ExprMetadata, ExpansionKind, node evaluation, `over`
                          # push-down (entry point is `Expr._with_over_node` in expr.py)
  _utils.py        # Implementation, Version, and shared helpers
  compliant.py     # Re-exports of the protocols that backends must implement
  plugins.py       # Plugin system for external backends
  dataframe.py     # Public DataFrame/LazyFrame API
  series.py        # Public Series API
  expr.py          # Public Expr API
  sql.py           # Public `narwhals.sql` (SQL generation; requires DuckDB)
  translate.py     # from_native / to_native (re-exported via narwhals/__init__.py)
  _translate.py    # Conversion/structural-typing protocols (NOT from_native/to_native)
  dependencies.py  # Backend detection (isinstance checks without imports)
tests/             # Test suite
tpch/              # TPC-H benchmark queries
packages/          # uv workspace members (currently: test-plugin)
```

## Hard rules

These are non-negotiable and the most common source of review comments. The long form, with
rationale, is in [CONTRIBUTING.md](CONTRIBUTING.md).

* **Zero dependencies.** Narwhals must never add a runtime dependency. It only uses what the user
  passes in.
* **Never import anything for `isinstance` checks.** Use the functions in
  [src/narwhals/dependencies.py](src/narwhals/dependencies.py) (e.g. `is_pandas_dataframe`).
* **Never iterate over rows.** Assume infinite rows. Column iteration is acceptable.
* **Never modify user input data.** Especially with pandas: no inplace operations on user-provided
  objects.
* **100% branch coverage** is enforced by the full-coverage CI job. When a branch is genuinely
  unreachable (e.g. gated on an unsupported backend version), mark it `# pragma: no cover` with a
  one-line reason.
* **Breaking changes never land in `narwhals.stable.v1` or `narwhals.stable.v2`.** New public APIs
  land in the main `narwhals` namespace and graduate into the next stable version. See
  [docs/backcompat.md](docs/backcompat.md), and add an entry to its `main` vs `stable.*` diff when
  the namespaces diverge.

Backend-specific rules (no pandas `apply`/`map`/`assign`/`drop`/`reset_index`/`rename`, no Polars
`map_elements`, no ordering assumptions or materialisation on lazy backends, DuckDB Python API over
SQL) are listed in full under
[CONTRIBUTING.md → Backend-specific considerations](CONTRIBUTING.md#backend-specific-considerations).

## The harness: run all of this before committing

Run these from the repo root. If you have not activated `.venv`, prefix the non-`make` commands with
`uv run`.

**1. Lint and formatting** (also runs the repo's custom checks: docstring validation, banned
imports, API-reference sync, slotted classes, `uv.lock` freshness):

```bash
prek run --all-files
```

**2. Static typing** (mypy, pyright, and pyrefly, per the `typing` target in
[Makefile](Makefile)):

```bash
make typing
```

Optionally, the type-completeness gate that CI also runs:

```bash
make typing-coverage
```

**3. Full test suite with 100% coverage** (this is the `pytest-full-coverage` CI job, and the one
that catches missing `# pragma: no cover`):

```bash
PYTEST_ADDOPTS="--numprocesses=logical" make run-ci DEPS="--extra pandas --extra dask --group core-tests --group sklearn --group plugins" CMD="pytest tests --cov=src --cov=tests --cov-fail-under=100 --runslow --durations=30 --constructors=pandas,pandas[nullable],pandas[pyarrow],pyarrow,polars[eager],polars[lazy],dask,duckdb,sqlframe"
```

**4. Doctests** (docstring examples are executed; reprs differ across versions, so CI only runs
these on the latest Python):

```bash
make run-ci DEPS="--extra pandas --extra dask --group core-tests --group sklearn" CMD="pytest src --doctest-modules"
```

**5. Docs build**, if you touched anything under `docs/` or any docstring. The build *executes* the
`exec="yes"` code blocks, so a stale snippet is a build failure:

```bash
make docs-build
```

To preview instead of just building: `make docs-serve` (or `make docs-clean-serve` if it does not
refresh). Docs are built with `zensical` (configured in [zensical.toml](zensical.toml)), not mkdocs
— the nav lives there, so a new page must be added to it.

### Faster inner loop

Full coverage runs are slow. While iterating:

```bash
uv run pytest tests/path/to/test_file.py            # one file
uv run pytest tests --constructors=pandas,polars[eager],pyarrow
uv run pytest tests --all-cpu-constructors          # needs --extra modin --extra pyspark
```

* Default constructors are `pandas,pandas[pyarrow],polars[eager],pyarrow,duckdb,sqlframe,ibis`
  (overridable via the `NARWHALS_DEFAULT_CONSTRUCTORS` env var).
* Hypothesis tests are skipped unless you pass `--runslow`.
* Dask and Modin are not in `local-dev`; add `--extra dask --extra modin` to test them locally.

Do not treat a green fast run as sufficient: the coverage gate and the lazy/SQL constructors
(`duckdb`, `sqlframe`, `dask`) catch a distinct class of bug, so run step 3 before committing.

### Test failure patterns

* `request.applymarker(pytest.mark.xfail)` — planned but not yet supported features.
* `pytest.mark.skipif` — conditional skips (e.g. version constraints).
* `pytest.raises` — expected exceptions.

Always document the reason in a comment. Details and examples:
[CONTRIBUTING.md → Test Failure Patterns](CONTRIBUTING.md#test-failure-patterns).

## Code style

* Line length: 90 characters. ruff for both formatting and linting.
* Docstrings: Google style, validated by `utils/check_docstrings.py` and darglint via `prek`.
  Docstring examples should import *one* dataframe library, and we deliberately balance which
  backend is used across the docs. See
  [CONTRIBUTING.md → Writing the doc(strings)](CONTRIBUTING.md#8-writing-the-docstrings).
* Static typing with mypy (strict), pyright, and pyrefly.
* In `_pandas_like/`, native types are typed as pandas types (the package is shared with Modin and
  cuDF). In `_spark_like/`, native types are typed as SQLFrame (shared with PySpark).
* Public API changes must be reflected in `docs/api-reference/` — `prek` fails otherwise.

## Pull requests

Title must start with a [conventional commit](https://www.conventionalcommits.org/) type: `build`,
`chore`, `ci`, `depr`, `docs`, `feat`, `fix`, `perf`, `refactor`, `release`, `test` (append `!` for
breaking changes). The title becomes the changelog entry.

**AI-assisted contributions must be disclosed** in the dedicated PR-template field, and the author
is accountable for every line. Read
[CONTRIBUTING.md → AI-assisted contributions](CONTRIBUTING.md#ai-assisted-contributions) before
opening a PR.
