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

## Code structure and design

Narwhals follows a layered design:

1. **Public API** ([src/narwhals/dataframe.py](src/narwhals/dataframe.py),
   [series.py](src/narwhals/series.py), [expr.py](src/narwhals/expr.py), ...): the user-facing
   Polars-like API. Thin wrappers that build `ExprNode`s and dispatch to compliant backends.
2. **Compliant wrappers**: one folder per backend (`_pandas_like/`, `_arrow/`, `_polars/`, `_duckdb/`,
   `_spark_like/`, `_dask/`, `_ibis/`), each backend implements Narwhals-compliant DataFrames,
   Series, Exprs, and Namespaces that translate the Polars-like API to native calls. Shared
   protocols and base classes live in `src/narwhals/_compliant/`.
3. **Native libraries** (pandas, Polars, PyArrow, ...): the actual computation engines, never
   directly depended on.

Outside `src/`, the project has:

- [tests/](tests/): test suite, see [CONTRIBUTING.md → Running tests](CONTRIBUTING.md#7-running-tests)
- [tpch/](tpch/): TPC-H benchmark queries, see [tpch/README.md](tpch/README.md)
- `packages/`: uv workspace members, built and installed independently of `src/narwhals`. Currently just a `test-plugin` that is a minimal fake backend used to test `narwhals.plugins`.

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

## Verify changes before committing

Run these from the repo root. If you have not activated `.venv`, prefix the non-`make` commands with
`uv run`.

1. `prek run --all-files` — lint, format, docstring/import checks
2. `make typing` — static type checking (mypy, pyright, and pyrefly)
   (Optional: `make typing-coverage` for type-completeness)
3. `make test-full-coverage` — full test suite with 100% coverage. Very slow, see [Faster testing](#faster-testing) for alternatives.
4. `make doctest` — tests docstring examples
5. `make docs-build` — run only if you touched `docs/` or a docstring. The build *executes* `exec="yes"`
   code blocks, so a stale snippet is a build failure. Docs are built with `zensical` (configured in
   [zensical.toml](zensical.toml)), not mkdocs — the nav lives there, so a new page must be added to it.

### Faster testing

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

## Inline Comments

Default to none. Only add one when the *why* is non-obvious
(a hidden constraint, a workaround, a subtle invariant). Never to restate what the code
already says, and never to narrate a diff. Comments that merely repeat the following
line are a common tell of unreviewed AI-assisted output and are removed on sight in review.

## Pull requests and issues

**A human opens the pull request or issue, not you.** Draft the diff, the description, and the
commit messages if asked, but the submission is made by the person you are working with, under their
account, in their words, after they have read the whole diff. The same holds for review replies and
issue discussion. Unattended agent-filed issues and PRs can be closed without discussion.

For guidelines about PRs (e.g. title rules) see
[CONTRIBUTING.md → Pull requests](CONTRIBUTING.md#6-pull-requests).

**AI-assisted contributions must be disclosed** in the dedicated PR-template field, and the author
is accountable for every line. Read
[CONTRIBUTING.md → AI-assisted contributions](CONTRIBUTING.md#ai-assisted-contributions) before
opening a PR.
